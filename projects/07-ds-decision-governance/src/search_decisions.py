"""
CLI: rank past decisions by similarity to a new proposal, so a decision
about to be written can check whether something like it has already
been decided, the natural question once a team has been logging
decisions for a while. Ranking is TF-IDF + cosine similarity over each
record's title, artifact_type, domain, and its "Why" section.

Run:
    python src/search_decisions.py "should we retire an old model or endpoint nobody uses anymore"
"""
import argparse
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schema import DecisionRecord, parse_decision_file

BASE = Path(__file__).resolve().parents[1]


def _why_section(body: str) -> str:
    """Pull the '## Why' section out of a decision's body, if present."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "## why":
            rest = "\n".join(lines[i + 1:])
            return rest.split("\n## ")[0].strip()
    return ""


def _record_text(record: DecisionRecord, body: str) -> str:
    return " ".join([record.title, record.artifact_type, record.domain, _why_section(body)])


def load_decisions(decisions_dir: Path) -> list[tuple[Path, DecisionRecord, str]]:
    """Parse every *.md file under decisions_dir into a (path, record, body) triple."""
    records = []
    for path in sorted(decisions_dir.rglob("*.md")):
        frontmatter, body = parse_decision_file(path)
        records.append((path, DecisionRecord(**frontmatter), body))
    return records


def search(
    query: str,
    records: list[tuple[Path, DecisionRecord, str]],
    top_n: int = 5,
) -> list[tuple[Path, DecisionRecord, float]]:
    """Rank records by cosine similarity of their TF-IDF vector to the
    query's. Returns the top_n (path, record, score) triples, highest
    score first."""
    if not records:
        return []
    texts = [_record_text(record, body) for _, record, body in records]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([*texts, query])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    ranked = sorted(zip(records, scores), key=lambda pair: -pair[1])
    return [(path, record, float(score)) for (path, record, _), score in ranked[:top_n]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Proposal or symptom description to search for.")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    records = load_decisions(BASE / "decisions")
    ranked = search(args.query, records, top_n=args.top)

    if not ranked:
        print("No decisions logged yet.")
        return

    print(f'Top {len(ranked)} match(es) for: "{args.query}"\n')
    for path, record, score in ranked:
        print(f"{score:.3f}  {record.id}  [{record.status}]  {record.title}")
        print(f"          {record.artifact_type} / {record.domain}")
        print(f"          {path.relative_to(BASE)}")
        print()


if __name__ == "__main__":
    sys.exit(main())
