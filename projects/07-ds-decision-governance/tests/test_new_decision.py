"""Unit tests for the record-scaffolding CLI: it fills in the template
correctly, assigns the next id, and the id/domain/title/author
substitutions land in the right places -- against a temp copy of the
real templates dir, not the real decisions/ directory."""
import shutil
from pathlib import Path

from new_decision import new_decision, next_id, slugify

BASE = Path(__file__).resolve().parents[1]


def _templates_dir(tmp_path):
    dst = tmp_path / "templates"
    shutil.copytree(BASE / "templates", dst)
    return dst


def test_slugify():
    assert slugify("Launch v3 churn-risk model!") == "launch-v3-churn-risk-model"


def test_next_id_starts_at_one_for_empty_dir(tmp_path):
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    assert next_id(decisions) == "DSG-0001"


def test_next_id_increments_past_existing(tmp_path):
    decisions = tmp_path / "decisions" / "product_analytics"
    decisions.mkdir(parents=True)
    (decisions / "a.md").write_text("---\nid: DSG-0003\n---\n")
    (decisions / "b.md").write_text("---\nid: DSG-0007\n---\n")
    assert next_id(tmp_path / "decisions") == "DSG-0008"


def test_new_decision_fills_in_template(tmp_path):
    path = new_decision(
        domain="marketing", impact_level="medium", title="Test the campaign dashboard",
        author="J. Doe",
        templates_dir=_templates_dir(tmp_path), decisions_dir=tmp_path / "decisions",
    )
    text = path.read_text()
    assert "id: DSG-0001" in text
    assert 'title: "Test the campaign dashboard"' in text
    assert "domain: marketing" in text
    assert 'author: "J. Doe"' in text
    assert "impact_level: medium" in text
    assert "YYYY-MM-DD" not in text
    assert path.parent.name == "marketing"


def test_new_decision_uses_next_available_id(tmp_path):
    decisions_dir = tmp_path / "decisions"
    templates_dir = _templates_dir(tmp_path)
    new_decision(domain="marketing", impact_level="low", title="First",
                 author="A", templates_dir=templates_dir, decisions_dir=decisions_dir)
    second = new_decision(domain="marketing", impact_level="low", title="Second",
                           author="A", templates_dir=templates_dir, decisions_dir=decisions_dir)
    assert "id: DSG-0002" in second.read_text()
