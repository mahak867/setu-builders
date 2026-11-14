"""Hard rules from the problem statement, checked against the real code paths."""
import hashlib
import hmac
import json

import pytest

from app import db, engine

CAT = engine.M["cats"]
NO_SLA = dict(recurrence=False, breach_probability=0.1, similar_in_60_days=0)


def draft(cat_id="CM-01", missing=()):
    return dict(missing=list(missing), category_id=cat_id)


# ---------------------------------------------------------------- policy gate: one test per rule
def test_sensitive_always_human():
    dec, rule, _ = engine.policy_gate(draft("CM-07"), CAT["CM-07"], 0.99, [], [], NO_SLA)
    assert (dec, rule) == ("HUMAN_REVIEW", "CM-SENS")


def test_sensitive_wins_over_everything_else():
    # high confidence, nothing missing, no flags: still never auto
    for cid in ("CM-06", "CM-07"):
        assert engine.policy_gate(draft(cid), CAT[cid], 1.0, [], [], NO_SLA)[0] == "HUMAN_REVIEW"


def test_injection_goes_to_human():
    assert engine.policy_gate(draft(), CAT["CM-01"], 0.99, ["ignore previous"], [], NO_SLA)[1] == "SAFETY-INJ"


def test_sensitive_cue_goes_to_human():
    assert engine.policy_gate(draft(), CAT["CM-01"], 0.99, [], ["scared"], NO_SLA)[1] == "SAFETY-CUE"


def test_low_confidence_goes_to_human():
    assert engine.policy_gate(draft(), CAT["CM-01"], 0.5, [], [], NO_SLA)[1] == "CONF-LOW"


def test_missing_details_goes_to_human():
    assert engine.policy_gate(draft(missing=["location"]), CAT["CM-01"], 0.99, [], [], NO_SLA)[1] == "INCOMPLETE"


def test_recurrence_escalates():
    sla = dict(recurrence=True, breach_probability=0.1, similar_in_60_days=5)
    assert engine.policy_gate(draft(), CAT["CM-01"], 0.99, [], [], sla)[1] == "ESC"


def test_breach_risk_escalates():
    sla = dict(recurrence=False, breach_probability=0.9, similar_in_60_days=0)
    assert engine.policy_gate(draft(), CAT["CM-01"], 0.99, [], [], sla)[1] == "ESC"


def test_field_work_category_needs_human():
    assert engine.policy_gate(draft("CM-02"), CAT["CM-02"], 0.99, [], [], NO_SLA)[1] == "CM-MANUAL"


def test_routine_case_auto_proposes():
    assert engine.policy_gate(draft(), CAT["CM-01"], 0.99, [], [], NO_SLA) [:2] == ("AUTO_PROPOSE", "AUTO-OK")


def test_every_decision_explains_itself():
    reason = engine.policy_gate(draft("CM-07"), CAT["CM-07"], 0.99, [], [], NO_SLA)[2]
    assert "CM-07" in reason and "Sensitive" in reason


# ---------------------------------------------------------------- state machine
def _case(text="भाग्य नगर मधील पथदिवा चार दिवसांपासून बंद आहे."):
    return engine.submit(text)


def test_no_rejected_state_exists():
    states = set(engine.ALLOWED) | {s for v in engine.ALLOWED.values() for s in v}
    assert not any("reject" in s.lower() for s in states)


def test_ai_can_never_close():
    g = _case()
    g = engine.department_callback(g["ref"], "done")
    with pytest.raises(engine.RuleViolation):
        engine.transition(g, "Closed", engine.AI)


def test_ai_cannot_propose_resolution_of_sensitive_case():
    g = engine.submit("वजीराबाद कार्यालय का बाबू फाइल के लिए रिश्वत मांग रहा है।")
    with pytest.raises(engine.RuleViolation):
        engine.transition(g, "Proposed resolved", engine.AI)


def test_department_callback_does_nothing_on_sensitive_case():
    g = engine.submit("वजीराबाद कार्यालय का बाबू फाइल के लिए रिश्वत मांग रहा है।")
    assert engine.department_callback(g["ref"], "done")["status"] == "In review"


def test_invalid_transition_rejected():
    g = _case()
    with pytest.raises(engine.RuleViolation):
        engine.transition(g, "Closed", "OFFICER:x")  # Assigned -> Closed skips citizen confirmation


def test_citizen_confirm_closes():
    g = engine.department_callback(_case()["ref"], "Lamp replaced")
    assert g["status"] == "Proposed resolved"
    assert engine.citizen_action(g["ref"], "confirm")["status"] == "Closed"


def test_citizen_dispute_reopens_automatically_and_raises_priority():
    g = engine.department_callback(_case()["ref"], "Lamp replaced")
    p = g["priority"]
    g = engine.citizen_action(g["ref"], "dispute")
    assert g["status"] == "Reopened" and g["reopened"] == 1 and g["priority"] > p


def test_closed_case_can_be_reopened_by_dispute():
    g = engine.department_callback(_case()["ref"], "done")
    engine.citizen_action(g["ref"], "confirm")
    assert engine.citizen_action(g["ref"], "dispute")["status"] == "Reopened"


# ---------------------------------------------------------------- audit and notifications
def test_every_ai_action_logged_with_required_fields():
    g = _case()
    rows = db.q("select * from audit where ref=? and actor='AI_SYSTEM'", (g["ref"],))
    assert len(rows) >= 5
    for r in rows:
        assert r["model"] and r["ts"] and r["reasoning"]
    assert any(r["rule_id"] == "AUTO-OK" and r["confidence"] for r in rows)


def test_citizen_notified_in_own_language():
    g = engine.submit("भाग्य नगर मधील पथदिवा चार दिवसांपासून बंद आहे.")
    body = db.q("select body from notification where ref=?", (g["ref"],))[0]["body"]
    assert "तक्रार" in body  # Marathi template


def test_webhook_is_hmac_signed():
    g = _case()
    w = db.q("select * from webhook where ref=?", (g["ref"],))[0]
    assert hmac.compare_digest(w["signature"], hmac.new(engine.HMAC_KEY, w["payload"].encode(), hashlib.sha256).hexdigest())
    assert json.loads(w["payload"])["ref"] == g["ref"]


def test_escalation_alerts_supervisor():
    g = engine.submit("तरोडा मधील गटार पुन्हा तुंबले आहे, सांडपाणी रस्त्यावर येत आहे.")
    assert g["rule_id"] == "ESC"
    assert db.q("select * from webhook where ref=? and event='escalation.alert'", (g["ref"],))
    assert any("ESCALATION" in n["body"] for n in db.q("select * from notification where ref=?", (g["ref"],)))


# ---------------------------------------------------------------- duplicates and priority
def test_duplicates_grouped_never_merged():
    refs = [engine.submit(t)["ref"] for t in ("इतवारा मधील रस्त्यावर मोठे खड्डे पडले आहेत.",
                                             "Huge potholes on the main road in Itwara, two bikes fell yesterday.",
                                             "इतवारा की सड़क पर गड्ढे हैं, बहुत परेशानी हो रही है।")]
    gs = [db.load_g(r) for r in refs]
    assert len(set(refs)) == 3
    assert len({g["group_id"] for g in gs}) == 1 and gs[0]["group_id"]
    for r in refs:  # each keeps its own updates
        assert db.q("select * from notification where ref=?", (r,))


def test_priority_reflects_severity_not_volume():
    garbage = [engine.submit("Garbage has been lying uncollected in Shivaji Nagar since Monday.") for _ in range(5)]
    wire = engine.submit("A live wire is hanging low near the school in Vazirabad, very dangerous.")
    assert wire["priority"] > max(g["priority"] for g in garbage)


def test_identity_hidden_for_identity_hiding_category():
    g = engine.submit("वजीराबाद कार्यालय का बाबू रिश्वत मांग रहा है। मेरा नाम सुरेश पाटिल है", name="Suresh Patil", phone="9876543210")
    m = engine.masked(g)
    assert "Suresh" not in m["name"] and m["phone"] == "Hidden" and "सुरेश" not in m["text"]


# ---------------------------------------------------------------- redaction
@pytest.mark.parametrize("raw,leak", [
    ("call me on 9876543210", "9876543210"),
    ("call +91 98765 43210", "98765 43210"),
    ("aadhaar 1234 5678 9012", "1234 5678 9012"),
    ("mail ravi.k@gmail.com", "ravi.k@gmail.com"),
    ("My name is Ravi Kumar", "Ravi Kumar"),
    ("मेरा नाम सुरेश पाटिल है", "सुरेश"),
    ("माझे नाव राहुल जाधव आहे", "राहुल"),
])
def test_redaction_removes_pii(raw, leak):
    assert leak not in engine.redact(raw)


@pytest.mark.parametrize("raw", ["मेरा नाम गुप्त रखें", "माझे नाव गुप्त ठेवा", "I am scared to go back", "please keep my name secret"])
def test_redaction_keeps_meaning(raw):
    assert engine.redact(raw) == raw
