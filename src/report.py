"""Generate a human-readable Markdown report from classified results."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_classified() -> List[Dict[str, str]]:
    """Load classified records from the CSV output."""
    import csv

    path = RESULTS_DIR / "classified.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src.scorer.py first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_summary() -> Dict[str, Any]:
    """Load the JSON summary."""
    path = RESULTS_DIR / "summary.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _examples(records: List[Dict[str, str]], category: str, classification: str, n: int = 2) -> str:
    """Return a markdown-formatted list of example prompts and responses."""
    filtered = [
        r for r in records
        if r["category"] == category and r["classification"] == classification
    ]
    if not filtered:
        return "_No examples in this category._\n"

    lines: List[str] = []
    for record in filtered[:n]:
        prompt = record["prompt"].strip()
        response = record["response"].strip().replace("\n", " ")
        lines.append(f"- **Prompt:** {prompt}\n")
        lines.append(f"  - **Response:** {response[:250]}{'...' if len(response) > 250 else ''}\n")
    return "\n".join(lines).strip()


def build_report(records: List[Dict[str, str]], summary: Dict[str, Any]) -> str:
    """Build a Markdown report string from results."""
    overall = summary["overall"]
    total = summary["total"]
    per_category = summary["per_category"]
    rates = summary["rates"]

    lines: List[str] = []
    lines.append("# LLM Safety Evaluation Report\n")
    lines.append(f"**Total prompts evaluated:** {total}\n")
    lines.append("\n## Overall Classification Counts\n")
    lines.append("| Classification | Count | Rate |")
    lines.append("|---|---|---|")
    for label in ["refused", "complied", "ambiguous"]:
        count = overall.get(label, 0)
        rate = rates["overall"].get(label, 0.0)
        lines.append(f"| {label.capitalize()} | {count} | {rate:.2%} |")

    lines.append("\n## Per-Category Results\n")
    lines.append("| Category | Refused | Complied | Ambiguous | Total |")
    lines.append("|---|---|---|---|---|")
    for category in sorted(per_category.keys()):
        counts = per_category[category]
        cat_total = sum(counts.values())
        refused = counts.get("refused", 0)
        complied = counts.get("complied", 0)
        ambiguous = counts.get("ambiguous", 0)
        lines.append(
            f"| {category} | {refused} | {complied} | {ambiguous} | {cat_total} |"
        )

    lines.append("\n## Per-Category Refusal Rates\n")
    lines.append("| Category | Refusal Rate |")
    lines.append("|---|---|")
    for category in sorted(per_category.keys()):
        rate = rates["per_category"][category].get("refused", 0.0)
        lines.append(f"| {category} | {rate:.2%} |")

    lines.append("\n## Selected Examples\n")
    for category in sorted(per_category.keys()):
        lines.append(f"\n### {category}\n")
        for classification in ["refused", "complied", "ambiguous"]:
            lines.append(f"**{classification.capitalize()} examples**\n")
            lines.append(_examples(records, category, classification))

    lines.append("\n## Notes and Limitations\n")
    lines.append(
        "- This report uses a simple keyword-based classifier. Borderline responses may be misclassified.\n"
    )
    lines.append(
        "- The mock responses (if used) are deterministic and intended to demonstrate the pipeline, not to evaluate a real model.\n"
    )
    lines.append(
        "- A stronger evaluation would include an LLM-as-judge, confidence intervals, and repeated runs across multiple models.\n"
    )

    return "\n".join(lines).strip()


def main() -> None:
    """Load classified results and write a Markdown report."""
    parser = argparse.ArgumentParser(description="Generate a Markdown report.")
    parser.parse_args()

    records = load_classified()
    summary = load_summary()
    report = build_report(records, summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "report.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
