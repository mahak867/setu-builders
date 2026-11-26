"""Loads the official HackFusion 4.0 datasets (data/*.csv) and validates them.

The Category Master is the single source of truth for categories, departments, SLAs and the Sensitive /
Identity-Hiding / auto-resolution flags. Nothing about categories is hard-coded in the engine: point
NYAY_DATA_DIR at another folder with the same three files and the whole system follows it.

Two small files are Nyay Setu's own configuration, not policy:
  lexicon.csv  multilingual terms per category_id, used only by the rules fallback when no LLM is available
  places.csv   Marathi / Hindi spellings of talukas and localities, used to recognise places in Devanagari text
"""
import csv
import os
from collections import Counter, defaultdict
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("NYAY_DATA_DIR", os.path.join(ROOT, "data"))

REQUIRED = {
    "grievance_category_master.csv": ["category_id", "category", "subcategory", "responsible_department_id", "responsible_department",
                                      "default_assignee", "sla_days", "sensitive", "identity_hiding", "default_priority", "auto_resolution_allowed"],
    "mock_grievance_history.csv": ["grievance_id", "complaint_text", "category_id", "department_id", "broad_location_taluka",
                                   "broad_location_locality", "submission_date", "resolution_date", "status", "resolution_text",
                                   "citizen_feedback", "sla_days", "sensitive", "identity_hiding", "resolution_actor"],
    "department_directory.csv": ["department_id", "department_name", "jurisdiction_mapping", "contact_email_demo", "contact_channel", "mock_webhook_id"],
}
SEVERITY = {"LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}


def _read(name, data_dir):
    with open(os.path.join(data_dir, name), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _bool(v):
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def load(data_dir=None):
    data_dir = data_dir or DATA_DIR
    issues, raw = [], {}
    for name, cols in REQUIRED.items():
        rows = _read(name, data_dir)
        missing = [c for c in cols if rows and c not in rows[0]]
        if missing:
            raise ValueError(f"{name}: missing required columns {missing}")
        raw[name] = rows

    depts = {}
    for d in raw["department_directory.csv"]:
        if d["department_id"] in depts:
            issues.append(f"duplicate department_id {d['department_id']}")
        depts[d["department_id"]] = dict(id=d["department_id"], name=d["department_name"], jurisdiction=d["jurisdiction_mapping"],
                                         email=d["contact_email_demo"], channel=d["contact_channel"], webhook_id=d["mock_webhook_id"],
                                         supervisor=f"{d['department_name']} - Supervisor (Demo)")
    cats = {}
    for c in raw["grievance_category_master.csv"]:
        if c["category_id"] in cats:
            issues.append(f"duplicate category_id {c['category_id']}")
        if c["responsible_department_id"] not in depts:
            issues.append(f"{c['category_id']}: department {c['responsible_department_id']} not in directory")
        try:
            sla = int(c["sla_days"])
        except ValueError:
            issues.append(f"{c['category_id']}: sla_days not an integer"); sla = 7
        prio = c["default_priority"].strip().upper()
        if prio not in SEVERITY:
            issues.append(f"{c['category_id']}: unknown default_priority {prio}")
        cats[c["category_id"]] = dict(id=c["category_id"], category=c["category"], subcategory=c["subcategory"],
                                      department_id=c["responsible_department_id"], department=c["responsible_department"],
                                      assignee=c["default_assignee"], sla_days=sla, sensitive=int(_bool(c["sensitive"])),
                                      identity_hiding=int(_bool(c["identity_hiding"])), priority=prio, severity=SEVERITY.get(prio, 3),
                                      auto_ok=int(_bool(c["auto_resolution_allowed"])), policy_note=c.get("policy_note", ""), keywords=[])
        if cats[c["category_id"]]["sensitive"] and cats[c["category_id"]]["auto_ok"]:
            issues.append(f"{c['category_id']}: sensitive but auto_resolution_allowed=True (sensitive wins in the policy gate)")

    hist, seen = [], set()
    for i, h in enumerate(raw["mock_grievance_history.csv"], 1):
        if h["grievance_id"] in seen:
            issues.append(f"duplicate grievance_id {h['grievance_id']}")
        seen.add(h["grievance_id"])
        if h["category_id"] not in cats:
            issues.append(f"{h['grievance_id']}: unknown category {h['category_id']}"); continue
        sub = h["submission_date"]
        res = h["resolution_date"] or None
        days = (date.fromisoformat(res) - date.fromisoformat(sub)).days if res else None
        sla = int(h["sla_days"] or cats[h["category_id"]]["sla_days"])
        hist.append(dict(id=i, grievance_id=h["grievance_id"], text=h["complaint_text"], text_en=h["complaint_text"], lang="en",
                         category_id=h["category_id"], department_id=h["department_id"], taluka=h["broad_location_taluka"],
                         locality=h["broad_location_locality"], submitted=sub, resolved=res, days=days, status=h["status"],
                         resolution=h["resolution_text"] or "(not yet resolved)", feedback=h["citizen_feedback"], sla_days=sla,
                         breached=int(days is not None and days > sla), actor=h["resolution_actor"],
                         sensitive=int(_bool(h["sensitive"]))))
        if _bool(h["sensitive"]) and _bool(h.get("ai_auto_closed", "False")):
            issues.append(f"{h['grievance_id']}: sensitive case marked ai_auto_closed in history")

    lex = defaultdict(list)
    if os.path.exists(os.path.join(data_dir, "lexicon.csv")):
        for r in _read("lexicon.csv", data_dir):
            if r["category_id"] in cats:
                lex[r["category_id"]] = [t.strip() for t in r["terms"].split("|") if t.strip()]
    for cid in cats:
        cats[cid]["keywords"] = lex.get(cid, [])

    places = {}  # locality or taluka (English) -> dict(taluka, locality, names=[...])
    loc_taluka = Counter((h["locality"], h["taluka"]) for h in hist)
    for (loc, tal), _ in loc_taluka.most_common():
        places.setdefault(loc, dict(taluka=tal, locality=loc, names=[loc]))
    for tal in {h["taluka"] for h in hist}:
        places.setdefault(tal, dict(taluka=tal, locality=None, names=[tal]))
    if os.path.exists(os.path.join(data_dir, "places.csv")):
        for r in _read("places.csv", data_dir):
            key = r["locality"] or r["taluka"]
            p = places.setdefault(key, dict(taluka=r["taluka"], locality=r["locality"] or None, names=[key]))
            p["names"] += [n for n in (r.get("mr"), r.get("hi")) if n]

    health = dict(data_dir=os.path.relpath(data_dir, ROOT) if data_dir.startswith(ROOT) else data_dir,
                  categories=len(cats), sensitive_categories=sum(c["sensitive"] for c in cats.values()),
                  identity_hiding_categories=sum(c["identity_hiding"] for c in cats.values()),
                  auto_resolution_categories=sum(c["auto_ok"] for c in cats.values()), departments=len(depts), history=len(hist),
                  talukas=len({h["taluka"] for h in hist}), localities=len({h["locality"] for h in hist}),
                  categories_without_lexicon=[c for c in cats if not cats[c]["keywords"]], issues=issues, ok=not issues)
    return dict(cats=cats, depts=depts, hist=hist, places=places, health=health)
