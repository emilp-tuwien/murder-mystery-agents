from __future__ import annotations

"""Compute judge-vs-human agreement (Cohen's kappa) for deception labels.

Consumes the CSV produced by analysis/export_deception_validation_sample.py
after a human annotator fills in the `human_label` column:

  - leave `human_label` BLANK        -> row is skipped (not yet annotated)
  - "none" / "not_deceptive" / "no"  -> human says NOT deceptive
  - a strategy label (or several,
    separated by , ; or |)           -> human says deceptive with these strategies
    (labels must come from schemas.deception.CANONICAL_STRATEGY_LABELS;
     unknown labels are warned about but still count as "deceptive")

Reported metrics (row level — each row is one judge label instance):
  - binary agreement: percent agreement, Cohen's kappa,
    sensitivity / specificity / precision treating the human as reference
  - strategy agreement (rows where BOTH say deceptive): exact-match rate of the
    judge's strategy against the human's label set, and multi-class Cohen's
    kappa over primary labels
  - optional second annotator column -> human-human binary kappa
    (inter-annotator reliability)

Usage:

    python analysis/compute_judge_agreement.py \\
        --input outputs/validation_sample_annotated.csv \\
        [--human-column human_label] [--second-human-column human_label_2] \\
        [--output outputs/judge_agreement.json]
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.deception import CANONICAL_STRATEGY_LABELS


NOT_DECEPTIVE_VALUES = {
    "none", "not_deceptive", "non_deceptive", "nondeceptive", "not deceptive",
    "no", "false", "0", "n", "honest", "clean",
}

TRUTHY = {"1", "true", "yes", "y"}


def _normalize_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def parse_human_label(raw: Any) -> Optional[Tuple[bool, List[str]]]:
    """Parse a human annotation cell.

    Returns None if the cell is blank (unannotated), otherwise
    (is_deceptive, [strategy_labels]).
    """
    text = str(raw or "").strip()
    if not text:
        return None
    normalized = _normalize_label(text)
    if normalized in NOT_DECEPTIVE_VALUES:
        return (False, [])
    parts = [
        _normalize_label(part)
        for chunk in text.split("|")
        for sub in chunk.split(";")
        for part in sub.split(",")
        if part.strip()
    ]
    labels = [p for p in parts if p and p not in NOT_DECEPTIVE_VALUES]
    return (bool(labels), labels)


def cohens_kappa(pairs: Sequence[Tuple[Any, Any]]) -> Optional[float]:
    """Cohen's kappa for paired categorical ratings."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    expected = sum(counts_a[k] * counts_b.get(k, 0) for k in counts_a) / (n * n)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def compute_agreement(
    rows: List[Dict[str, Any]],
    human_column: str = "human_label",
    second_human_column: Optional[str] = None,
) -> Dict[str, Any]:
    annotated: List[Dict[str, Any]] = []
    unknown_labels: Counter = Counter()

    for row in rows:
        parsed = parse_human_label(row.get(human_column))
        if parsed is None:
            continue
        human_deceptive, human_labels = parsed
        for label in human_labels:
            if label not in CANONICAL_STRATEGY_LABELS:
                unknown_labels[label] += 1
        judge_deceptive = _as_bool(row.get("is_deceptive_instance"))
        judge_label = _normalize_label(str(row.get("strategy_label") or "none"))
        annotated.append({
            "judge_deceptive": judge_deceptive,
            "judge_label": judge_label,
            "human_deceptive": human_deceptive,
            "human_labels": human_labels,
            "row": row,
        })

    n = len(annotated)
    result: Dict[str, Any] = {
        "total_rows_in_file": len(rows),
        "annotated_rows": n,
        "skipped_unannotated_rows": len(rows) - n,
        "unknown_human_labels": dict(unknown_labels),
    }
    if n == 0:
        result["error"] = f"No rows have a non-blank '{human_column}' value."
        return result

    # ── Binary agreement ────────────────────────────────────────────────────
    binary_pairs = [(a["judge_deceptive"], a["human_deceptive"]) for a in annotated]
    tp = sum(1 for j, h in binary_pairs if j and h)
    tn = sum(1 for j, h in binary_pairs if not j and not h)
    fp = sum(1 for j, h in binary_pairs if j and not h)
    fn = sum(1 for j, h in binary_pairs if not j and h)

    result["binary"] = {
        "confusion_matrix": {
            "judge_yes_human_yes": tp,
            "judge_no_human_no": tn,
            "judge_yes_human_no": fp,
            "judge_no_human_yes": fn,
        },
        "percent_agreement": (tp + tn) / n,
        "cohens_kappa": cohens_kappa(binary_pairs),
        # Human treated as reference standard:
        "sensitivity": tp / (tp + fn) if (tp + fn) else None,
        "specificity": tn / (tn + fp) if (tn + fp) else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "judge_positive_rate": (tp + fp) / n,
        "human_positive_rate": (tp + fn) / n,
    }

    # ── Strategy agreement (both deceptive) ─────────────────────────────────
    both_deceptive = [a for a in annotated if a["judge_deceptive"] and a["human_deceptive"]]
    if both_deceptive:
        in_set = sum(1 for a in both_deceptive if a["judge_label"] in a["human_labels"])
        strategy_pairs = [
            (a["judge_label"], a["human_labels"][0]) for a in both_deceptive if a["human_labels"]
        ]
        result["strategy"] = {
            "rows_both_deceptive": len(both_deceptive),
            "judge_label_in_human_set_rate": in_set / len(both_deceptive),
            "primary_label_exact_match_rate": (
                sum(1 for j, h in strategy_pairs if j == h) / len(strategy_pairs)
                if strategy_pairs else None
            ),
            "primary_label_cohens_kappa": cohens_kappa(strategy_pairs),
            "judge_label_distribution": dict(Counter(a["judge_label"] for a in both_deceptive)),
            "human_primary_label_distribution": dict(
                Counter(a["human_labels"][0] for a in both_deceptive if a["human_labels"])
            ),
        }
    else:
        result["strategy"] = {"rows_both_deceptive": 0}

    # ── Optional second annotator (inter-annotator reliability) ─────────────
    if second_human_column:
        pairs: List[Tuple[bool, bool]] = []
        for a in annotated:
            parsed2 = parse_human_label(a["row"].get(second_human_column))
            if parsed2 is None:
                continue
            pairs.append((a["human_deceptive"], parsed2[0]))
        result["inter_annotator"] = {
            "column_a": human_column,
            "column_b": second_human_column,
            "double_annotated_rows": len(pairs),
            "percent_agreement": (
                sum(1 for x, y in pairs if x == y) / len(pairs) if pairs else None
            ),
            "binary_cohens_kappa": cohens_kappa(pairs),
        }

    return result


def _format_report(result: Dict[str, Any]) -> str:
    lines = ["Judge vs human agreement", "=" * 40]
    lines.append(f"Annotated rows: {result.get('annotated_rows')} / {result.get('total_rows_in_file')}")
    if result.get("unknown_human_labels"):
        lines.append(f"WARNING — unknown human labels: {result['unknown_human_labels']}")
    binary = result.get("binary")
    if binary:
        kappa = binary.get("cohens_kappa")
        lines.append("")
        lines.append("Binary (deceptive vs not):")
        lines.append(f"  percent agreement: {binary['percent_agreement']:.3f}")
        lines.append(f"  Cohen's kappa:     {kappa:.3f}" if kappa is not None else "  Cohen's kappa:     n/a")
        lines.append(f"  sensitivity:       {binary['sensitivity']:.3f}" if binary.get("sensitivity") is not None else "  sensitivity:       n/a")
        lines.append(f"  specificity:       {binary['specificity']:.3f}" if binary.get("specificity") is not None else "  specificity:       n/a")
        lines.append(f"  precision:         {binary['precision']:.3f}" if binary.get("precision") is not None else "  precision:         n/a")
        cm = binary["confusion_matrix"]
        lines.append(
            f"  confusion: TP={cm['judge_yes_human_yes']} TN={cm['judge_no_human_no']} "
            f"FP={cm['judge_yes_human_no']} FN={cm['judge_no_human_yes']}"
        )
    strategy = result.get("strategy", {})
    if strategy.get("rows_both_deceptive"):
        lines.append("")
        lines.append(f"Strategy (n={strategy['rows_both_deceptive']} rows both deceptive):")
        lines.append(f"  judge label in human set: {strategy['judge_label_in_human_set_rate']:.3f}")
        if strategy.get("primary_label_exact_match_rate") is not None:
            lines.append(f"  primary exact match:      {strategy['primary_label_exact_match_rate']:.3f}")
        if strategy.get("primary_label_cohens_kappa") is not None:
            lines.append(f"  primary Cohen's kappa:    {strategy['primary_label_cohens_kappa']:.3f}")
    inter = result.get("inter_annotator")
    if inter:
        lines.append("")
        lines.append(f"Inter-annotator ({inter['column_a']} vs {inter['column_b']}, n={inter['double_annotated_rows']}):")
        if inter.get("percent_agreement") is not None:
            lines.append(f"  percent agreement: {inter['percent_agreement']:.3f}")
        if inter.get("binary_cohens_kappa") is not None:
            lines.append(f"  binary kappa:      {inter['binary_cohens_kappa']:.3f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Cohen's kappa between the LLM deception judge and human annotations."
    )
    parser.add_argument("--input", required=True, help="Annotated validation-sample CSV.")
    parser.add_argument("--human-column", default="human_label", help="Column holding human annotations (default: human_label).")
    parser.add_argument("--second-human-column", default=None, help="Optional second annotator column for inter-annotator kappa.")
    parser.add_argument("--output", default=None, help="Path for the JSON report (default: <input>.agreement.json).")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    result = compute_agreement(rows, human_column=args.human_column, second_human_column=args.second_human_column)
    result["input_path"] = str(input_path)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".agreement.json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(_format_report(result))
    print(f"\nJSON report written to: {output_path}")


if __name__ == "__main__":
    main()
