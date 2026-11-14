"""The LLM leads intake but can only make the system more cautious. Tested with a fake LLM (no network)."""
import os

import pytest
from fastapi.testclient import TestClient

from app import engine

BRIBE = "वजीराबाद कार्यालय का बाबू मेरी फाइल आगे बढ़ाने के लिए रिश्वत मांग रहा है।"
LIGHT = "भाग्य नगर मधील पथदिवा चार दिवसांपासून बंद आहे."


def fake_llm(monkeypatch, **over):
    base = dict(model="fake-llm", lang="mr", category_id="CM-01", locality="Bhagya Nagar", urgency=2, affected=1, sentiment="neutral",
                missing=[], possibly_sensitive=False, sensitive_reason=None, contains_instructions=False,
                english="The streetlight in Bhagya Nagar has been off for four days.", summary="Streetlight off in Bhagya Nagar.", follow_up=None)
    base.update(over)
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setattr(engine, "llm_intake", lambda text: dict(base))


# ---------------------------------------------------------------- schema validation
def test_invalid_llm_values_are_dropped():
    v = engine.validate_llm(dict(lang="fr", category_id="CM-99", locality="Atlantis", urgency=42, affected="lots", sentiment="happy",
                                 missing=["location", "everything"], possibly_sensitive="yes"), "m")
    assert v["lang"] is None and v["category_id"] is None and v["locality"] is None
    assert v["urgency"] == 5 and v["affected"] is None and v["sentiment"] is None
    assert v["missing"] == ["location"] and v["possibly_sensitive"] is False  # only a real boolean counts


def test_llm_error_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setattr(engine, "llm_intake", lambda t: dict(error="HTTP 503"))
    g = engine.submit(LIGHT)
    assert g["engine"].startswith("rules") and g["decision"] == "AUTO_PROPOSE"


def test_llm_led_routine_case(monkeypatch):
    fake_llm(monkeypatch)
    g = engine.submit(LIGHT)
    assert g["engine"] == "llm+rules" and g["decision"] == "AUTO_PROPOSE" and g["confidence_parts"]["llm_rules_agree"] == 1


# ---------------------------------------------------------------- the LLM can only add caution
def test_llm_cannot_downgrade_a_sensitive_case(monkeypatch):
    fake_llm(monkeypatch, category_id="CM-01", lang="hi")  # LLM wrongly says "streetlight"
    g = engine.submit(BRIBE)
    assert g["category_id"] == "CM-07" and g["decision"] == "HUMAN_REVIEW" and g["rule_id"] == "CM-SENS"


def test_llm_can_upgrade_to_sensitive(monkeypatch):
    fake_llm(monkeypatch, category_id="CM-06")
    g = engine.submit("Bhagya Nagar office staff keeps calling me at night.")
    assert g["sensitive"] and g["decision"] == "HUMAN_REVIEW"


def test_llm_sensitivity_flag_routes_to_human(monkeypatch):
    fake_llm(monkeypatch, possibly_sensitive=True, sensitive_reason="implied quid pro quo")
    g = engine.submit(LIGHT)
    assert g["rule_id"] == "SAFETY-CUE" and g["identity_hidden"]


def test_llm_instruction_flag_routes_to_human(monkeypatch):
    fake_llm(monkeypatch, contains_instructions=True)
    assert engine.submit(LIGHT)["rule_id"] == "SAFETY-INJ"


def test_disagreement_lowers_confidence(monkeypatch):
    fake_llm(monkeypatch)
    agree = engine.submit(LIGHT)["confidence"]
    fake_llm(monkeypatch, category_id="CM-04")
    disagree = engine.submit(LIGHT)
    assert disagree["confidence"] < agree and disagree["decision"] == "HUMAN_REVIEW"


def test_llm_location_needs_to_be_a_known_place(monkeypatch):
    fake_llm(monkeypatch, locality=None)
    g = engine.submit("The streetlight outside my house is not working.")
    assert "location" in g["missing"] and g["decision"] == "HUMAN_REVIEW"


def test_llm_follow_up_question_used_in_chat(monkeypatch):
    fake_llm(monkeypatch, locality=None, missing=["location"], follow_up="ही समस्या कुठे आहे?")
    r = engine.chat("s1", "आमच्या घरासमोरील पथदिवा बंद आहे.")
    assert not r["complete"] and r["reply"] == "ही समस्या कुठे आहे?"


def test_cross_language_retrieval_uses_translation():
    hits = engine.retrieve("इतवारा की सड़क पर गड्ढे हैं", "There are potholes on the road in Itwara.")
    assert {h["lang"] for h in hits} - {"hi"} and all(h["category_id"] == "CM-05" for h in hits)


# ---------------------------------------------------------------- API
@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_api_master_data(client):
    assert len(client.get("/api/master/categories").json()) == 10
    assert client.get("/api/master/departments").status_code == 200


def test_api_chat_asks_for_location(client):
    r = client.post("/api/chat", json=dict(session_id="a", message="The streetlight outside my house is not working.")).json()
    assert not r["complete"] and "where" in r["reply"].lower()


def test_api_full_flow_submit_track_dispute(client):
    client.post("/api/chat", json=dict(session_id="b", message=LIGHT))
    g = client.post("/api/grievances", json=dict(session_id="b")).json()
    assert g["status"] == "Assigned"
    client.post(f"/api/webhooks/department-callback/{g['ref']}", json=dict(action="done", note="fixed"))
    assert client.get(f"/api/grievances/{g['ref']}").json()["status"] == "Proposed resolved"
    assert client.post(f"/api/grievances/{g['ref']}/citizen", json=dict(action="dispute")).json()["status"] == "Reopened"


def test_api_blocks_invalid_transition(client):
    g = client.post("/api/grievances", json=dict(text=LIGHT)).json()
    assert client.post(f"/api/grievances/{g['ref']}/officer", json=dict(action="close")).status_code == 409


def test_api_officer_view_masks_identity(client):
    g = client.post("/api/grievances", json=dict(text=BRIBE + " मेरा नाम सुरेश पाटिल है", name="Suresh", phone="9876543210")).json()
    d = client.get(f"/api/grievances/{g['ref']}?role=officer").json()
    assert d["phone"] == "Hidden" and "सुरेश" not in d["text"]


def test_api_transcribe_without_key_reports_fallback(client):
    assert "error" in client.post("/api/transcribe?lang=mr", content=b"\x00" * 10, headers={"Content-Type": "audio/webm"}).json()
