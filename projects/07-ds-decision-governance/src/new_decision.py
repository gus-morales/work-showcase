"""
CLI: scaffold a new decision record from the right template.

Run:
    python src/new_decision.py --domain product_analytics --impact-level medium \\
        --title "Switch the onboarding funnel dashboard to weekly cohorts" --author "J. Okafor"
"""
import argparse
import re
import sys
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ID_PATTERN = re.compile(r"^id:\s*DSG-(\d+)", re.MULTILINE)


def next_id(decisions_dir: Path) -> str:
    max_n = 0
    for path in decisions_dir.rglob("*.md"):
        match = ID_PATTERN.search(path.read_text())
        if match:
            max_n = max(max_n, int(match.group(1)))
    return f"DSG-{max_n + 1:04d}"


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60]


def new_decision(domain: str, impact_level: str, title: str, author: str,
                  templates_dir: Path = None, decisions_dir: Path = None) -> Path:
    templates_dir = templates_dir or (BASE / "templates")
    decisions_dir = decisions_dir or (BASE / "decisions")

    template_path = templates_dir / f"decision-{impact_level}.md"
    if not template_path.exists():
        raise ValueError(f"no template for impact_level={impact_level}")

    text = template_path.read_text()
    record_id = next_id(decisions_dir)

    text = re.sub(r"^id:\s*DSG-XXXX", f"id: {record_id}", text, flags=re.MULTILINE)
    text = re.sub(r'^title:\s*""', f'title: "{title}"', text, flags=re.MULTILINE)
    text = re.sub(r"^domain:\s*\S+(\s*#.*)?", f"domain: {domain}", text, flags=re.MULTILINE)
    text = re.sub(r'^author:\s*""', f'author: "{author}"', text, flags=re.MULTILINE)
    text = re.sub(r"^(\s*proposed:)\s*YYYY-MM-DD", rf"\1 {date.today().isoformat()}", text, flags=re.MULTILINE)

    out_dir = decisions_dir / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record_id}-{slugify(title)}.md"
    out_path.write_text(text)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--impact-level", required=True, choices=["low", "medium", "high"])
    parser.add_argument("--title", required=True)
    parser.add_argument("--author", default="")
    args = parser.parse_args()

    path = new_decision(args.domain, args.impact_level, args.title, args.author)
    print(f"Wrote {path.relative_to(BASE)}")
    print("Fill in the remaining fields (dates, reviewers, body sections) before opening a PR.")


if __name__ == "__main__":
    sys.exit(main())
