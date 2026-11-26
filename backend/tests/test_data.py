"""The Category Master CSV is the single source of truth: swap the files and behaviour follows, with no code change."""
import csv
import os
import shutil

import pytest

from app import data, engine

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL = os.path.join(ROOT, "data")


def test_official_dataset_loads_cleanly():
    h = data.load(OFFICIAL)["health"]
    assert h["ok"] and h["categories"] == 30 and h["departments"] == 20 and h["history"] == 800 and h["talukas"] == 16
    assert h["sensitive_categories"] == 5 and h["categories_without_lexicon"] == []


def test_every_category_routes_to_a_known_department():
    d = data.load(OFFICIAL)
    assert all(c["department_id"] in d["depts"] for c in d["cats"].values())


def test_missing_required_column_is_rejected(tmp_path):
    for f in os.listdir(OFFICIAL):
        shutil.copy(os.path.join(OFFICIAL, f), tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "grievance_category_master.csv", encoding="utf-8-sig")))
    cols = [c for c in rows[0] if c != "sensitive"]
    with open(tmp_path / "grievance_category_master.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError):
        data.load(str(tmp_path))


@pytest.fixture
def swapped(tmp_path):
    """Copy the official files and flip flags in the Category Master: C009 streetlights become Sensitive."""
    for f in os.listdir(OFFICIAL):
        shutil.copy(os.path.join(OFFICIAL, f), tmp_path)
    p = tmp_path / "grievance_category_master.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for r in rows:
        if r["category_id"] == "C009":
            r["sensitive"], r["identity_hiding"], r["auto_resolution_allowed"] = "True", "True", "False"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    engine.bootstrap(str(tmp_path))
    yield
    engine.bootstrap(OFFICIAL)


def test_csv_swap_changes_policy_without_code_change(swapped):
    g = engine.submit("Two streetlights on the lane near Itwara are switched off.")
    assert g["category_id"] == "C009" and g["rule_id"] == "CM-SENS" and g["identity_hidden"]


def test_official_flags_restored_after_swap():
    g = engine.submit("Two streetlights on the lane near Itwara are switched off.")
    assert g["category_id"] == "C009" and not g["sensitive"]
