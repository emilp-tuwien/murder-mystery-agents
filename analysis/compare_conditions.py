from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import argparse
import csv
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.metrics import aggregate_experiment_conditions


METRIC_SPECS = [
    {
        "key": "group_solve_rate",
        "label": "Group solve rate (RQ3)",
        "higher_is_better_for_detection": True,
        "baseline_key": "random_group_solve_rate_baseline",
    },
    {
        "key": "mean_murderer_vote_share",
        "label": "Mean murderer vote share (RQ3)",
        "higher_is_better_for_detection": True,
        "baseline_key": "random_vote_share_baseline",
    },
    {
        "key": "mean_structured_accusation_fraction",
        "label": "Structured accusation fraction (thesis quality)",
        "higher_is_better_for_detection": True,
        "baseline_key": None,
    },
    {
        "key": "thesis_usable_rate",
        "label": "Thesis-usable run rate (workflow quality)",
        "higher_is_better_for_detection": True,
        "baseline_key": None,
    },
    {
        "key": "mean_murderer_attention_received",
        "label": "Mean murderer attention received (RQ2)",
        "higher_is_better_for_detection": True,
        "baseline_key": None,
    },
    {
        "key": "mean_accusation_confidence",
        "label": "Mean accusation confidence (decision quality)",
        "higher_is_better_for_detection": True,
        "baseline_key": None,
    },
    {
        "key": "mean_murderer_followups_received",
        "label": "Mean murderer follow-up questions received (RQ2)",
        "higher_is_better_for_detection": True,
        "baseline_key": None,
    },
    {
        "key": "mean_murderer_justification_requests_received",
        "label": "Mean murderer justification requests received (RQ2)",
        "higher_is_better_for_detection": True,
        "baseline_key": None,
    },
    {
        "key": "mean_pressure_target_gini",
        "label": "Pressure concentration Gini (RQ2)",
        "higher_is_better_for_detection": False,
        "baseline_key": None,
    },
    {
        "key": "mean_murderer_speaker_share",
        "label": "Mean murderer speaker share (RQ2)",
        "higher_is_better_for_detection": False,
        "baseline_key": None,
    },
    {
        "key": "mean_murderer_labeled_utterances",
        "label": "Mean labeled murderer utterances (RQ1)",
        "higher_is_better_for_detection": False,
        "baseline_key": None,
    },
]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any, digits: int = 3) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> Dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0}

    phat = successes / total
    denominator = 1 + (z ** 2 / total)
    center = (phat + z ** 2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1 - phat) / total) + (z ** 2 / (4 * total * total)))
        / denominator
    )
    return {
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def _two_proportion_z_test(success_a: int, total_a: int, success_b: int, total_b: int) -> Dict[str, float | None]:
    if total_a <= 0 or total_b <= 0:
        return {"z": None, "p_value": None}

    p1 = success_a / total_a
    p2 = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    variance = pooled * (1 - pooled) * ((1 / total_a) + (1 / total_b))
    if variance <= 0:
        return {"z": None, "p_value": None}

    z = (p1 - p2) / math.sqrt(variance)
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    return {"z": z, "p_value": p_value}


def _load_condition_rows(experiment_dir: Path) -> List[Dict[str, Any]]:
    summary = aggregate_experiment_conditions(experiment_dir)
    return list(summary.get("conditions", []))


def _rank_conditions(condition_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []

    for spec in METRIC_SPECS:
        key = spec["key"]
        rows_with_values = [row for row in condition_rows if _safe_float(row.get(key)) is not None]
        reverse = bool(spec["higher_is_better_for_detection"])
        ordered = sorted(rows_with_values, key=lambda row: float(row[key]), reverse=reverse)
        ranked.append(
            {
                "metric": key,
                "label": spec["label"],
                "ranking": [
                    {
                        "rank": index,
                        "condition_name": row.get("condition_name"),
                        "value": row.get(key),
                    }
                    for index, row in enumerate(ordered, start=1)
                ],
            }
        )

    return ranked


def _baseline_assessment(condition_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assessments: List[Dict[str, Any]] = []

    for row in condition_rows:
        condition_name = row.get("condition_name")
        total_runs = int(row.get("total_runs") or 0)
        group_solve_rate = _safe_float(row.get("group_solve_rate")) or 0.0
        baseline = _safe_float(row.get("random_group_solve_rate_baseline")) or 0.0
        solved_runs = int(round(group_solve_rate * total_runs)) if total_runs else 0
        expected_random_solves = baseline * total_runs if total_runs else 0.0
        ci = _wilson_interval(solved_runs, total_runs) if total_runs else {"lower": 0.0, "upper": 0.0}
        test = _two_proportion_z_test(solved_runs, total_runs, int(round(expected_random_solves)), total_runs) if total_runs else {"z": None, "p_value": None}

        assessments.append(
            {
                "condition_name": condition_name,
                "total_runs": total_runs,
                "observed_group_solves": solved_runs,
                "observed_group_solve_rate": group_solve_rate,
                "observed_group_solve_rate_ci95": ci,
                "chance_group_solve_rate": baseline,
                "expected_random_solves": expected_random_solves,
                "lift_over_chance": group_solve_rate - baseline,
                "z_statistic_vs_chance": test.get("z"),
                "p_value_vs_chance": test.get("p_value"),
            }
        )

    return assessments


def _pairwise_differences(condition_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comparisons: List[Dict[str, Any]] = []
    by_name = {row.get("condition_name"): row for row in condition_rows}
    ordered_names = [row.get("condition_name") for row in condition_rows if row.get("condition_name")]

    for index, left_name in enumerate(ordered_names):
        for right_name in ordered_names[index + 1 :]:
            left = by_name[left_name]
            right = by_name[right_name]

            row: Dict[str, Any] = {
                "condition_a": left_name,
                "condition_b": right_name,
                "total_runs_a": left.get("total_runs"),
                "total_runs_b": right.get("total_runs"),
            }

            for spec in METRIC_SPECS:
                key = spec["key"]
                left_value = _safe_float(left.get(key))
                right_value = _safe_float(right.get(key))
                row[f"delta__{key}"] = (left_value - right_value) if left_value is not None and right_value is not None else None

            solve_rate_a = _safe_float(left.get("group_solve_rate")) or 0.0
            solve_rate_b = _safe_float(right.get("group_solve_rate")) or 0.0
            total_a = int(left.get("total_runs") or 0)
            total_b = int(right.get("total_runs") or 0)
            solves_a = int(round(solve_rate_a * total_a)) if total_a else 0
            solves_b = int(round(solve_rate_b * total_b)) if total_b else 0
            test = _two_proportion_z_test(solves_a, total_a, solves_b, total_b)
            row["group_solve_rate_z"] = test.get("z")
            row["group_solve_rate_p_value"] = test.get("p_value")
            comparisons.append(row)

    return comparisons


def build_condition_report(experiment_dir: Path) -> Dict[str, Any]:
    condition_rows = _load_condition_rows(experiment_dir)
    report = {
        "experiment_dir": str(experiment_dir),
        "total_conditions": len(condition_rows),
        "condition_rankings": _rank_conditions(condition_rows),
        "chance_baseline_assessment": _baseline_assessment(condition_rows),
        "pairwise_differences": _pairwise_differences(condition_rows),
    }
    return report


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_condition_report(experiment_dir: Path) -> Dict[str, Any]:
    report = build_condition_report(experiment_dir)
    report_path = experiment_dir / "condition_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    pairwise_rows = report.get("pairwise_differences", [])
    if pairwise_rows:
        _write_csv(experiment_dir / "condition_pairwise_differences.csv", pairwise_rows)

    baseline_rows = []
    for row in report.get("chance_baseline_assessment", []):
        baseline_rows.append(
            {
                "condition_name": row.get("condition_name"),
                "total_runs": row.get("total_runs"),
                "observed_group_solves": row.get("observed_group_solves"),
                "observed_group_solve_rate": row.get("observed_group_solve_rate"),
                "observed_group_solve_rate_ci95_lower": row.get("observed_group_solve_rate_ci95", {}).get("lower"),
                "observed_group_solve_rate_ci95_upper": row.get("observed_group_solve_rate_ci95", {}).get("upper"),
                "chance_group_solve_rate": row.get("chance_group_solve_rate"),
                "lift_over_chance": row.get("lift_over_chance"),
                "z_statistic_vs_chance": row.get("z_statistic_vs_chance"),
                "p_value_vs_chance": row.get("p_value_vs_chance"),
            }
        )
    if baseline_rows:
        _write_csv(experiment_dir / "condition_chance_baseline.csv", baseline_rows)

    summary_lines = [
        f"Condition report for {experiment_dir}",
        "",
        "Rankings:",
    ]
    for ranking in report.get("condition_rankings", []):
        summary_lines.append(f"- {ranking.get('label')}:")
        for item in ranking.get("ranking", []):
            summary_lines.append(
                f"  {item.get('rank')}. {item.get('condition_name')} = {_format_float(item.get('value'))}"
            )

    summary_lines.append("")
    summary_lines.append("Chance baseline checks (RQ3):")
    for row in report.get("chance_baseline_assessment", []):
        ci = row.get("observed_group_solve_rate_ci95", {})
        summary_lines.append(
            "- "
            f"{row.get('condition_name')}: observed={_format_float(row.get('observed_group_solve_rate'))} "
            f"(95% CI {_format_float(ci.get('lower'))}–{_format_float(ci.get('upper'))}), "
            f"chance={_format_float(row.get('chance_group_solve_rate'))}, "
            f"lift={_format_float(row.get('lift_over_chance'))}, "
            f"p={_format_float(row.get('p_value_vs_chance'))}"
        )

    markdown_path = experiment_dir / "condition_report.md"
    markdown_path.write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Summarize and compare condition-level experiment outputs.")
    parser.add_argument("experiment_dir", help="Path to an experiment output directory.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    aggregate_summary = aggregate_experiment_conditions(experiment_dir)
    report = write_condition_report(experiment_dir)
    print(json.dumps({"aggregate_summary": aggregate_summary, "condition_report": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
