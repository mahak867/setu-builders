"""Hard rules from the problem statement and the official dataset guide, checked against the real code paths."""
import hashlib
import hmac
import json

import pytest

from app import db, engine

CAT = engine.M["cats"]
NO_SLA = dict(recurrence=False, breach_probability=0.1, district_breach_rate=0.65, poor_outcomes_here=0, window_days=365, past_case_ids=[])
ROUTINE = "Two streetlights on the lane near Itwara are switched off."  # C009, auto_resolution_allowed=True
SENSITIVE = "A man follows and harasses women near the Kinwat bus stand every evening."  # C017, Sensitive=True


def draft(cat_id="C009", missing=(), location="Itwara"):
    return dict(missing=list(missing), category_id=cat_id, location=location)


# ---------------------------------------------------------------- policy gate: one test per rule
@pytest.mark.parametrize("cid", [c for c in CAT if CAT[c]["sensitive"]])
def test_every_sensitive_category_always_goes_to_a_human(cid):
    dec, rule, reason = engine.policy_gate(draft(cid), CAT[cid], 1.0, [], [], NO_SLA)
    assert (dec, rule) == ("HUMAN_REVIEW", "CM-SENS") and cid in reason


def test_injection_goes_to_human():
    assert engine.policy_gate(draft(), CAT["C009"], 0.99, ["ignore previous"], [], NO_SLA)[1] == "SAFETY-INJ"


def test_sensitive_cue_goes_to_human():
    assert engine.policy_gate(draft(), CAT["C009"], 0.99, [], ["bribe"], NO_SLA)[1] == "SAFETY-CUE"


def test_low_confidence_goes_to_human():
    assert engine.policy_gate(draft(), CAT["C009"], 0.5, [], [], NO_SLA)[1] == "CONF-LOW"


def test_missing_details_goes_to_human():
    assert engine.policy_gate(draft(missing=["location"]), CAT["C009"], 0.99, [], [], NO_SLA)[1] == "INCOMPLETE"


def test_recurrence_escalates():
    sla = dict(NO_SLA, recurrence=True, poor_outcomes_here=3, past_case_ids=["GRV-1", "GRV-2", "GRV-3"])
    assert engine.policy_gate(draft(), CAT["C009"], 0.99, [], [], sla)[1] == "ESC"


def test_breach_risk_escalates():
    assert engine.policy_gate(draft(), CAT["C009"], 0.99, [], [], dict(NO_SLA, breach_probability=0.95))[1] == "ESC"


def test_routine_case_auto_proposes():
    assert engine.policy_gate(draft(), CAT["C009"], 0.99, [], [], NO_SLA)[:2] == ("AUTO_PROPOSE", "AUTO-OK")


# ---------------------------------------------------------------- state machine
def test_no_rejected_state_exists():
    states = set(engine.ALLOWED) | {s for v in engine.ALLOWED.values() for s in v}
    assert not any("reject" in s.lower() for s in states)


def test_ai_can_never_close():
    g = engine.department_callback(engine.submit(ROUTINE)["ref"], "done")
    with pytest.raises(engine.RuleViolation):
        engine.transition(g, "Closed", engine.AI)


def test_ai_cannot_propose_resolution_of_sensitive_case():
    g = engine.submit(SENSITIVE)
    assert g["sensitive"]
    with pytest.raises(engine.RuleViolation):
        engine.transition(g, "Proposed resolved", engine.AI)


def test_department_callback_does_nothing_on_sensitive_case():
    g = engine.submit(SENSITIVE)
    assert engine.department_callback(g["ref"], "done")["status"] == "In review"


def test_invalid_transition_rejected():
    g = engine.submit(ROUTINE)
    with pytest.raises(engine.RuleViolation):
        engine.transition(g, "Closed", "OFFICER:x")  # Assigned -> Closed skips citizen confirmation


def test_citizen_confirm_closes():
    g = engine.department_callback(engine.submit(ROUTINE)["ref"], "Lamp replaced")
    assert g["status"] == "Proposed resolved"
    assert engine.citizen_action(g["ref"], "confirm")["status"] == "Closed"


def test_citizen_dispute_reopens_automatically_and_raises_priority():
    g = engine.department_callback(engine.submit(ROUTINE)["ref"], "Lamp replaced")
    p = g["priority"]
    g = engine.citizen_action(g["ref"], "dispute")
    assert g["status"] == "Reopened" and g["reopened"] == 1 and g["priority"] > p


def test_closed_case_can_be_reopened_by_dispute():
    g = engine.department_callback(engine.submit(ROUTINE)["ref"], "done")
    engine.citizen_action(g["ref"], "confirm")
    assert engine.citizen_action(g["ref"], "dispute")["status"] == "Reopened"


# ---------------------------------------------------------------- audit, routing, notifications
def test_every_ai_action_logged_with_required_fields():
    g = engine.submit(ROUTINE)
    rows = db.q("select * from audit where ref=? and actor='AI_SYSTEM'", (g["ref"],))
    assert len(rows) >= 5
    for r in rows:
        assert r["model"] and r["ts"] and r["reasoning"]
    assert any(r["rule_id"] == "AUTO-OK" and r["confidence"] for r in rows)


def test_routed_to_department_from_category_master_and_directory():
    g = engine.submit(ROUTINE)
    c = CAT[g["category_id"]]
    w = json.loads(db.q("select payload from webhook where ref=?", (g["ref"],))[0]["payload"])
    assert g["department"] == c["department"] and w["department_id"] == c["department_id"]
    assert w["mock_webhook_id"] == engine.M["depts"][c["department_id"]]["webhook_id"]


def test_due_date_uses_category_sla():
    g = engine.submit(ROUTINE)
    from datetime import datetime
    assert (datetime.fromisoformat(g["due"]) - datetime.fromisoformat(g["created"])).days == CAT[g["category_id"]]["sla_days"]


def test_citizen_notified_in_own_language():
    g = engine.submit("वजीराबाद कार्यालयात उत्पन्नाचा दाखला अर्ज दिला होता, अजून मिळाला नाही.")
    body = db.q("select body from notification where ref=?", (g["ref"],))[0]["body"]
    assert "तक्रार" in body  # Marathi template


def test_webhook_is_hmac_signed():
    g = engine.submit(ROUTINE)
    w = db.q("select * from webhook where ref=?", (g["ref"],))[0]
    assert hmac.compare_digest(w["signature"], hmac.new(engine.HMAC_KEY, w["payload"].encode(), hashlib.sha256).hexdigest())


def test_recurring_problem_escalates_with_evidence():
    g = engine.submit("पाथरड येथे पुन्हा कचरा उचलला नाही, घंटागाडी आलीच नाही.")
    assert g["rule_id"] == "ESC" and g["sla"]["past_case_ids"]
    assert db.q("select * from webhook where ref=? and event='escalation.alert'", (g["ref"],))
    assert any("ESCALATION" in n["body"] for n in db.q("select * from notification where ref=?", (g["ref"],)))


def test_bribery_is_a_sensitivity_signal_even_without_a_category():
    g = engine.submit("Bhokar Town office clerk is asking for a bribe to process my ration card.")
    assert g["decision"] == "HUMAN_REVIEW" and g["identity_hidden"]


# ---------------------------------------------------------------- duplicates and priority
def test_duplicates_suggested_never_merged_and_officer_confirms():
    refs = [engine.submit(t)["ref"] for t in ("इतवारा मधील रस्त्यावर मोठे खड्डे पडले आहेत.",
                                             "Huge potholes on the main road in Itwara, two bikes fell yesterday.",
                                             "इतवारा की सड़क पर गड्ढे हैं, बहुत परेशानी हो रही है।")]
    gs = [db.load_g(r) for r in refs]
    assert len(set(refs)) == 3 and len({g["group_id"] for g in gs}) == 1 and gs[0]["group_id"]
    assert {g["group_status"] for g in gs} == {"suggested"}
    engine.officer_action(refs[0], "confirm_group", "")
    assert {db.load_g(r)["group_status"] for r in refs} == {"confirmed"}
    for r in refs:
        assert db.q("select * from notification where ref=?", (r,))


def test_rejected_group_is_not_suggested_again():
    a = engine.submit("Huge potholes on the main road in Itwara, two bikes fell yesterday.")
    b = engine.submit("इतवारा मधील रस्त्यावर मोठे खड्डे पडले आहेत.")
    engine.officer_action(b["ref"], "reject_group", "different roads")
    c = engine.submit("इतवारा की सड़क पर गड्ढे हैं।")
    assert c["group_id"] is None and a["ref"] != c["ref"]


def test_priority_reflects_severity_not_volume():
    garbage = [engine.submit("Garbage has been lying uncollected in Shivaji Nagar since Monday.") for _ in range(5)]
    flood = engine.submit("Flood water has entered houses in Loha Town after the storm, families need help.")
    assert flood["priority"] > max(g["priority"] for g in garbage)


def test_identity_hidden_for_identity_hiding_category():
    g = engine.submit(SENSITIVE + " My name is Sunita Jadhav, phone 9876543210", name="Sunita Jadhav", phone="9876543210")
    m = engine.masked(g)
    assert "Sunita" not in m["name"] and m["phone"] == "Hidden" and "9876543210" not in m["text"]


def test_public_trace_view_hides_precise_locality():
    pv = engine.public_view({"text": "Streetlight off in Shivaji Nagar", "location": "Shivaji Nagar"})
    assert "Shivaji Nagar" not in json.dumps(pv) and "Nanded" in json.dumps(pv)


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


@pytest.mark.parametrize("term,text,hit", [("hiv", "Shivaji Nagar", False), ("harass", "harassment daily", True), ("park", "parking", True),
                                            ("गटार", "गटारे तुंबली", True)])
def test_keyword_matching_respects_word_starts(term, text, hit):
    assert engine.term_in(term, text) is hit
