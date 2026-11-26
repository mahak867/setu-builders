"""The LLM leads intake but can only make the system more cautious. Tested with a fake LLM (no network)."""
import os

import pytest
from fastapi.testclient import TestClient

from app import engine

BRIBE = "वजीराबाद कार्यालय का बाबू मेरी फाइल आगे बढ़ाने के लिए रिश्वत मांग रहा है।"
LIGHT = "Two streetlights on the lane near Itwara are switched off."
HARASS = "A man follows and harasses women near the Kinwat bus stand every evening."


def fake_llm(monkeypatch, **over):
    base = dict(model="fake-llm", lang="mr", category_id="C009", locality="Itwara", urgency=2, affected=1, sentiment="neutral",
                missing=[], possibly_sensitive=False, sensitive_reason=None, contains_instructions=False,
                english="Two streetlights on the lane near Itwara are switched off.", summary="Streetlights off in Itwara.", follow_up=None)
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
    fake_llm(monkeypatch, category_id="C009", lang="en", locality="Kinwat")  # LLM wrongly says "streetlight"
    g = engine.submit(HARASS)
    assert g["category_id"] == "C017" and g["decision"] == "HUMAN_REVIEW" and g["rule_id"] == "CM-SENS"


def test_llm_cannot_clear_a_bribery_signal(monkeypatch):
    fake_llm(monkeypatch, category_id="C025", lang="hi", locality="Vazirabad", possibly_sensitive=False)
    g = engine.submit(BRIBE)
    assert g["decision"] == "HUMAN_REVIEW" and g["identity_hidden"]


def test_llm_can_upgrade_to_sensitive(monkeypatch):
    fake_llm(monkeypatch, category_id="C017")
    g = engine.submit("Itwara office staff keeps calling me at night.")
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
    fake_llm(monkeypatch, category_id="C006")
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
    assert all(h["category_id"] == "C001" for h in hits)  # official history is English; the Hindi query still finds it


# ---------------------------------------------------------------- API
@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_api_master_data(client):
    assert len(client.get("/api/master/categories").json()) == 30 and len(client.get("/api/categories").json()) == 30
    assert client.get("/api/data/health").json()["ok"]
    assert client.get("/api/master/departments").status_code == 200


def test_api_chat_asks_for_location(client):
    r = client.post("/api/chat", json=dict(session_id="a", message="The streetlight outside my house is not working.")).json()  # no place named
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
    g = client.post("/api/grievances", json=dict(text=HARASS + " मेरा नाम सुरेश पाटिल है", name="Suresh", phone="9876543210")).json()
    d = client.get(f"/api/grievances/{g['ref']}?role=officer").json()
    assert d["phone"] == "Hidden" and "सुरेश" not in d["text"]


def test_api_transcribe_without_key_reports_fallback(client):
    assert "error" in client.post("/api/transcribe?lang=mr", content=b"\x00" * 10, headers={"Content-Type": "audio/webm"}).json()


def test_api_guide_endpoints(client):
    g = client.post("/api/grievances", json=dict(text=LIGHT)).json()
    assert client.get("/api/grievances/history?q=streetlight&k=3").json()[0]["category_id"] == "C009"
    assert client.get(f"/api/audit-logs?ref={g['ref']}&actor=AI_SYSTEM").json()
    assert client.post("/api/work-orders", json=dict(ref=g["ref"], note="inspect")).json()["ok"]
    assert client.patch(f"/api/grievances/{g['ref']}/status", json=dict(action="close")).status_code == 409  # cannot skip citizen confirmation
