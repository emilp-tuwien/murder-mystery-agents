from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv
import json
import random

from analysis.attention import build_attention_artifacts
from analysis.deception import aggregate_condition_deception, label_deception_for_run
from utils.dialogue_analysis import detect_direct_address, extract_mentions, is_question


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_events(events_path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not events_path.exists():
        return events
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _estimate_random_solve_rate(agent_names: List[str], murderer_name: str, trials: int = 5000, seed: int = 0) -> float:
    if not agent_names or murderer_name not in agent_names:
        return 0.0

    rng = random.Random(seed)
    solved = 0
    for _ in range(trials):
        votes = Counter()
        for accuser in agent_names:
            candidates = [name for name in agent_names if name != accuser]
            accused = rng.choice(candidates)
            votes[accused] += 1
        max_votes = max(votes.values()) if votes else 0
        winners = [name for name, count in votes.items() if count == max_votes]
        if murderer_name in winners:
            solved += 1
    return solved / trials if trials else 0.0


def _extract_turn_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    turn_rows: List[Dict[str, Any]] = []
    current_turn: Dict[str, Any] | None = None

    for event in events:
        payload = event.get("payload", {})
        event_type = event.get("type")

        if event_type == "turn_started":
            if current_turn is not None:
                turn_rows.append(current_turn)
            current_turn = {
                "turn": payload.get("turn"),
                "round": payload.get("round"),
                "phase": payload.get("phase"),
                "stage": payload.get("stage"),
                "selected_speaker": None,
                "selection_reason": None,
                "is_direct_address": False,
            }
        elif event_type == "speaker_selected" and current_turn is not None:
            current_turn["selected_speaker"] = payload.get("speaker")
            current_turn["selection_reason"] = payload.get("reason")
            current_turn["is_direct_address"] = payload.get("is_direct_address", False)

    if current_turn is not None:
        turn_rows.append(current_turn)

    return turn_rows


def _extract_utterance_rows(events: List[Dict[str, Any]], agent_names: List[str], run_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    current_round = 1
    current_phase = "introduction"
    current_stage = "introduction"

    for event in events:
        event_type = event.get("type")
        payload = event.get("payload", {})
        if event_type == "turn_started":
            current_round = payload.get("round", current_round)
            current_phase = payload.get("phase", current_phase)
            current_stage = payload.get("stage", current_stage)
        if event_type != "utterance":
            continue

        utterance = dict(payload.get("utterance", {}))
        speaker = utterance.get("speaker")
        text = utterance.get("text", "")
        other_agents = [name for name in agent_names if name != speaker]
        addressed_to = utterance.get("addressed_to") or detect_direct_address(text, other_agents)
        mentioned_agents = utterance.get("mentioned_agents") or extract_mentions(text, other_agents)
        question_flag = utterance.get("is_question")
        if question_flag is None:
            question_flag = is_question(text)

        rows.append({
            "run_id": run_id,
            "turn": utterance.get("turn"),
            "round": utterance.get("round", current_round),
            "phase": utterance.get("phase", current_phase),
            "stage": utterance.get("stage", current_stage),
            "speaker": speaker,
            "text": text,
            "word_count": len(text.split()),
            "is_question": question_flag,
            "addressed_to": addressed_to,
            "mentioned_agents": "|".join(mentioned_agents),
            "response_to_speaker": utterance.get("response_to_speaker"),
        })

    return rows


def _extract_accusation_rows(events: List[Dict[str, Any]], murderer_name: str, run_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in events:
        if event.get("type") != "accusation":
            continue
        payload = event.get("payload", {})
        agent = payload.get("agent")
        result = payload.get("result", {})
        accused = result.get("accused")
        evidence_items = result.get("evidence_items") or []
        belief_snapshot = payload.get("belief_snapshot", {}) or {}
        belief_alignment = payload.get("belief_alignment", {}) or {}
        ranking = belief_snapshot.get("ranking", []) or []
        top_n_candidates = belief_alignment.get("top_n_candidates") or [row.get("name") for row in ranking[:3] if row.get("name")]
        if isinstance(evidence_items, list):
            evidence_items_str = " | ".join(str(item).strip() for item in evidence_items if str(item).strip())
            evidence_item_count = len([item for item in evidence_items if str(item).strip()])
        else:
            evidence_items_str = str(evidence_items)
            evidence_item_count = len([part for part in str(evidence_items).split("|") if part.strip()])
        rows.append({
            "run_id": run_id,
            "accuser": agent,
            "accused": accused,
            "reasoning": result.get("reasoning", ""),
            "confidence": result.get("confidence"),
            "primary_basis": result.get("primary_basis", "mixed"),
            "evidence_items": evidence_items_str,
            "evidence_item_count": evidence_item_count,
            "motive_case": result.get("motive_case", ""),
            "means_case": result.get("means_case", ""),
            "opportunity_case": result.get("opportunity_case", ""),
            "contradiction_case": result.get("contradiction_case", ""),
            "comparative_case": result.get("comparative_case", ""),
            "uncertainty": result.get("uncertainty", ""),
            "belief_top_suspect": belief_alignment.get("top_suspect") or belief_snapshot.get("top_suspect"),
            "belief_uncertainty": belief_snapshot.get("uncertainty"),
            "belief_top_gap": belief_snapshot.get("top_gap"),
            "belief_accused_rank": belief_alignment.get("accused_rank"),
            "belief_accused_in_top_n": belief_alignment.get("accused_in_top_n"),
            "belief_top_n_candidates": "|".join(str(item) for item in top_n_candidates if item),
            "belief_corrected_to_top_suspect": belief_alignment.get("corrected_to_top_suspect", False),
            "belief_snapshot": json.dumps(belief_snapshot, ensure_ascii=False, sort_keys=True),
            "correct": accused == murderer_name,
            "accuser_is_murderer": agent == murderer_name,
            "accused_is_murderer": accused == murderer_name,
        })
    return rows


def _extract_belief_rows(events: List[Dict[str, Any]], run_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in events:
        if event.get("type") != "beliefs_updated":
            continue
        payload = event.get("payload", {}) or {}
        beliefs = payload.get("beliefs", {}) or {}
        for agent_name, snapshot in beliefs.items():
            ranking = snapshot.get("ranking", []) or []
            scores = snapshot.get("suspicion_scores", {}) or {}
            rows.append({
                "run_id": run_id,
                "turn": payload.get("turn", snapshot.get("turn")),
                "round": payload.get("round", snapshot.get("round")),
                "stage": payload.get("stage", snapshot.get("stage")),
                "context": snapshot.get("context", "post_utterance"),
                "observed_speaker": payload.get("observed_speaker", snapshot.get("observed_speaker")),
                "agent": agent_name,
                "top_suspect": snapshot.get("top_suspect"),
                "top_suspect_score": snapshot.get("top_suspect_score"),
                "top_gap": snapshot.get("top_gap"),
                "uncertainty": snapshot.get("uncertainty"),
                "top_1": ranking[0].get("name") if len(ranking) >= 1 else None,
                "top_2": ranking[1].get("name") if len(ranking) >= 2 else None,
                "top_3": ranking[2].get("name") if len(ranking) >= 3 else None,
                "ranking_json": json.dumps(ranking, ensure_ascii=False, sort_keys=True),
                "suspicion_scores_json": json.dumps(scores, ensure_ascii=False, sort_keys=True),
                "top_reasons_json": json.dumps(snapshot.get("top_reasons", {}), ensure_ascii=False, sort_keys=True),
            })
    return rows


def _compute_agent_metrics(utterances: List[Dict[str, Any]], agent_names: List[str], murderer_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    utterance_count = Counter()
    word_count = Counter()
    question_count = Counter()
    questions_received = Counter()
    mentions_received = Counter()
    question_edges = Counter()
    mention_edges = Counter()

    for row in utterances:
        speaker = row["speaker"]
        utterance_count[speaker] += 1
        word_count[speaker] += int(row["word_count"])
        if row["is_question"]:
            question_count[speaker] += 1
            if row["addressed_to"]:
                questions_received[row["addressed_to"]] += 1
                question_edges[(speaker, row["addressed_to"])] += 1

        mentioned = [name for name in row["mentioned_agents"].split("|") if name]
        for target in mentioned:
            mentions_received[target] += 1
            mention_edges[(speaker, target)] += 1

    total_utterances = sum(utterance_count.values()) or 1
    total_words = sum(word_count.values()) or 1

    agent_rows: List[Dict[str, Any]] = []
    for agent in agent_names:
        agent_rows.append({
            "agent": agent,
            "is_murderer": agent == murderer_name,
            "utterance_count": utterance_count[agent],
            "speaker_share": utterance_count[agent] / total_utterances,
            "word_count": word_count[agent],
            "word_share": word_count[agent] / total_words,
            "question_count": question_count[agent],
            "questions_received": questions_received[agent],
            "mentions_received": mentions_received[agent],
            "attention_received": questions_received[agent] + mentions_received[agent],
        })

    question_rows = [
        {"source": source, "target": target, "count": count}
        for (source, target), count in sorted(question_edges.items())
    ]
    mention_rows = [
        {"source": source, "target": target, "count": count}
        for (source, target), count in sorted(mention_edges.items())
    ]
    return agent_rows, question_rows, mention_rows


def _compute_accusation_quality(accusations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not accusations:
        return {
            "mean_confidence": 0.0,
            "structured_evidence_fraction": 0.0,
            "mean_evidence_item_count": 0.0,
            "motive_case_fraction": 0.0,
            "means_case_fraction": 0.0,
            "opportunity_case_fraction": 0.0,
            "contradiction_case_fraction": 0.0,
        }

    evidence_backed = 0
    evidence_item_total = 0
    motive_case_count = 0
    means_case_count = 0
    opportunity_case_count = 0
    contradiction_case_count = 0
    confidence_values = []

    for row in accusations:
        confidence_values.append(float(row.get("confidence") or 0))
        evidence_item_count = int(row.get("evidence_item_count") or 0)
        evidence_item_total += evidence_item_count
        if evidence_item_count >= 2:
            evidence_backed += 1
        if str(row.get("motive_case", "")).strip():
            motive_case_count += 1
        if str(row.get("means_case", "")).strip():
            means_case_count += 1
        if str(row.get("opportunity_case", "")).strip():
            opportunity_case_count += 1
        if str(row.get("contradiction_case", "")).strip():
            contradiction_case_count += 1

    total = len(accusations)
    return {
        "mean_confidence": sum(confidence_values) / total,
        "structured_evidence_fraction": evidence_backed / total,
        "mean_evidence_item_count": evidence_item_total / total,
        "motive_case_fraction": motive_case_count / total,
        "means_case_fraction": means_case_count / total,
        "opportunity_case_fraction": opportunity_case_count / total,
        "contradiction_case_fraction": contradiction_case_count / total,
    }


def _compute_rq3(accusations: List[Dict[str, Any]], agent_names: List[str], murderer_name: str) -> Dict[str, Any]:
    votes = Counter(row["accused"] for row in accusations if row.get("accused"))
    total_votes = len(accusations) or 1
    max_votes = max(votes.values()) if votes else 0
    winners = [name for name, count in votes.items() if count == max_votes] if max_votes else []
    murderer_vote_share = votes.get(murderer_name, 0) / total_votes
    group_solved = murderer_name in winners
    random_solve_rate = _estimate_random_solve_rate(agent_names, murderer_name)
    random_vote_share = 1 / len(agent_names) if agent_names else 0.0

    top1_alignment = 0
    topn_alignment = 0
    uncertainty_values: List[float] = []
    corrected_to_top = 0
    for row in accusations:
        accused_rank = row.get("belief_accused_rank")
        if accused_rank is not None and str(accused_rank).strip():
            try:
                accused_rank_int = int(float(accused_rank))
                if accused_rank_int == 1:
                    top1_alignment += 1
                if accused_rank_int <= 3:
                    topn_alignment += 1
            except (TypeError, ValueError):
                pass
        if row.get("belief_uncertainty") not in (None, ""):
            try:
                uncertainty_values.append(float(row.get("belief_uncertainty")))
            except (TypeError, ValueError):
                pass
        if str(row.get("belief_corrected_to_top_suspect", "")).strip().lower() in {"1", "true", "yes"}:
            corrected_to_top += 1

    return {
        "total_votes": len(accusations),
        "vote_counts": dict(votes),
        "murderer_vote_share": murderer_vote_share,
        "group_solved": group_solved,
        "winning_suspects": winners,
        "belief_top1_alignment_fraction": top1_alignment / total_votes,
        "belief_top3_alignment_fraction": topn_alignment / total_votes,
        "mean_belief_uncertainty": (sum(uncertainty_values) / len(uncertainty_values)) if uncertainty_values else 0.0,
        "belief_forced_top_suspect_count": corrected_to_top,
        # Escape metrics: RQ3 asks whether the murderer *avoids* accusation above chance.
        "murderer_escaped": not group_solved,
        "murderer_escape_rate_this_run": 0.0 if group_solved else 1.0,
        "random_vote_share_baseline": random_vote_share,
        "random_group_solve_rate_baseline": random_solve_rate,
        "random_escape_rate_baseline": 1.0 - random_solve_rate,
    }


def analyze_run(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    manifest = _read_json(run_path / "run_manifest.json")
    events = load_events(run_path / "events.jsonl")
    run_id = manifest.get("run_id", run_path.name)
    agent_names = manifest.get("agent_names", [])
    murderer_name = manifest.get("murderer_name")

    turn_rows = _extract_turn_rows(events)
    utterance_rows = _extract_utterance_rows(events, agent_names, run_id)
    accusation_rows = _extract_accusation_rows(events, murderer_name, run_id)
    belief_rows = _extract_belief_rows(events, run_id)
    agent_rows, question_rows, mention_rows = _compute_agent_metrics(utterance_rows, agent_names, murderer_name)
    attention = build_attention_artifacts(run_path, utterance_rows, agent_names, murderer_name)
    accusation_quality = _compute_accusation_quality(accusation_rows)
    rq3 = _compute_rq3(accusation_rows, agent_names, murderer_name)
    rq1_raw = label_deception_for_run(run_path, utterance_rows, manifest)
    rq1 = {
        "total_murderer_utterances": rq1_raw.get("total_murderer_utterances", 0),
        "labeled_murderer_utterances": rq1_raw.get("labeled_murderer_utterances", 0),
        "deceptive_instance_count": rq1_raw.get("deceptive_instance_count", rq1_raw.get("total_labeled_utterances", 0)),
        "proportion_murderer_utterances_deceptive": rq1_raw.get("proportion_murderer_utterances_deceptive", 0.0),
        "counts_by_strategy": rq1_raw.get("counts_by_strategy", rq1_raw.get("strategy_counts", {})),
        "rates_by_strategy": rq1_raw.get("rates_by_strategy", rq1_raw.get("strategy_rates", {})),
        "strategies_present": rq1_raw.get("strategies_present", sorted((rq1_raw.get("strategy_counts") or {}).keys())),
        "judge_method": rq1_raw.get("judge_method", "heuristic"),
        "judge_model": rq1_raw.get("judge_model", "n/a"),
        "judge_temperature": rq1_raw.get("judge_temperature", "n/a"),
        # Legacy keys for backward-compat with aggregation code
        "total_labeled_instances": rq1_raw.get("total_labeled_instances", 0),
        "total_labeled_utterances": rq1_raw.get("total_labeled_utterances", 0),
        "strategy_counts": rq1_raw.get("counts_by_strategy", rq1_raw.get("strategy_counts", {})),
        "strategy_rates": rq1_raw.get("rates_by_strategy", rq1_raw.get("strategy_rates", {})),
    }

    murderer_agent_row = next((row for row in agent_rows if row["agent"] == murderer_name), None)
    attention_summary = attention.get("summary", {})
    summary = {
        "run_id": run_id,
        "experiment_name": manifest.get("experiment_name"),
        "condition_name": manifest.get("condition_name"),
        "condition_description": manifest.get("condition_description"),
        "condition_factors": manifest.get("condition_factors", {}),
        "murderer_name": murderer_name,
        "total_turns": len(turn_rows),
        "total_utterances": len(utterance_rows),
        "total_belief_snapshots": len(belief_rows),
        "rq1": rq1,
        "rq2": {
            "murderer_speaker_share": murderer_agent_row["speaker_share"] if murderer_agent_row else 0.0,
            "murderer_questions_received": murderer_agent_row["questions_received"] if murderer_agent_row else 0,
            "murderer_mentions_received": murderer_agent_row["mentions_received"] if murderer_agent_row else 0,
            "murderer_attention_received": murderer_agent_row["attention_received"] if murderer_agent_row else 0,
            "murderer_followups_received": attention_summary.get("murderer_followups_received", 0),
            "murderer_justification_requests_received": attention_summary.get("murderer_justification_requests_received", 0),
            "murderer_pressure_signals_received": attention_summary.get("murderer_pressure_signals_received", 0),
            "question_target_entropy": attention_summary.get("question_target_entropy", 0.0),
            "question_target_gini": attention_summary.get("question_target_gini", 0.0),
            "pressure_target_entropy": attention_summary.get("pressure_target_entropy", 0.0),
            "pressure_target_gini": attention_summary.get("pressure_target_gini", 0.0),
            "followup_target_entropy": attention_summary.get("followup_target_entropy", 0.0),
            "followup_target_gini": attention_summary.get("followup_target_gini", 0.0),
            "justification_target_entropy": attention_summary.get("justification_target_entropy", 0.0),
            "justification_target_gini": attention_summary.get("justification_target_gini", 0.0),
        },
        "accusation_quality": accusation_quality,
        "rq3": rq3,
    }

    _write_csv(run_path / "turns.csv", turn_rows)
    _write_csv(run_path / "utterances.csv", utterance_rows)
    _write_csv(run_path / "accusations.csv", accusation_rows)
    _write_csv(run_path / "beliefs.csv", belief_rows)
    _write_csv(run_path / "agent_metrics.csv", agent_rows)
    _write_csv(run_path / "question_edges.csv", question_rows)
    _write_csv(run_path / "mention_edges.csv", mention_rows)

    with (run_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary


def aggregate_experiment(experiment_dir: str | Path) -> Dict[str, Any]:
    experiment_path = Path(experiment_dir)
    runs_dir = experiment_path / "runs"
    summaries: List[Dict[str, Any]] = []

    if runs_dir.exists():
        for run_path in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            metrics_path = run_path / "metrics.json"
            if metrics_path.exists():
                summaries.append(_read_json(metrics_path))

    if not summaries:
        return {"experiment_dir": str(experiment_path), "total_runs": 0}

    aggregate_rows = []
    solve_count = 0
    escape_count = 0
    usable_count = 0
    warning_count = 0
    for summary in summaries:
        rq3 = summary.get("rq3", {})
        run_id = summary.get("run_id")
        validation = _read_json(experiment_path / "runs" / str(run_id) / "run_validation.json") if run_id else {}
        if rq3.get("group_solved"):
            solve_count += 1
        if rq3.get("murderer_escaped"):
            escape_count += 1
        if validation.get("run_usable_for_thesis"):
            usable_count += 1
        if validation.get("warnings"):
            warning_count += 1
        aggregate_rows.append({
            "run_id": run_id,
            "condition_name": summary.get("condition_name"),
            "murderer_name": summary.get("murderer_name"),
            "total_turns": summary.get("total_turns"),
            "total_utterances": summary.get("total_utterances"),
            "total_belief_snapshots": summary.get("total_belief_snapshots", 0),
            "murderer_speaker_share": summary.get("rq2", {}).get("murderer_speaker_share"),
            "murderer_attention_received": summary.get("rq2", {}).get("murderer_attention_received"),
            "murderer_followups_received": summary.get("rq2", {}).get("murderer_followups_received"),
            "murderer_justification_requests_received": summary.get("rq2", {}).get("murderer_justification_requests_received"),
            "murderer_pressure_signals_received": summary.get("rq2", {}).get("murderer_pressure_signals_received"),
            "question_target_entropy": summary.get("rq2", {}).get("question_target_entropy"),
            "pressure_target_gini": summary.get("rq2", {}).get("pressure_target_gini"),
            "murderer_labeled_utterances": summary.get("rq1", {}).get("deceptive_instance_count", summary.get("rq1", {}).get("total_labeled_utterances")),
            "murderer_proportion_deceptive": summary.get("rq1", {}).get("proportion_murderer_utterances_deceptive", 0.0),
            "murderer_judge_method": summary.get("rq1", {}).get("judge_method", "heuristic"),
            "murderer_direct_denial_rate": summary.get("rq1", {}).get("rates_by_strategy", summary.get("rq1", {}).get("strategy_rates", {})).get("direct_denial", 0.0),
            "murderer_deflection_rate": summary.get("rq1", {}).get("rates_by_strategy", summary.get("rq1", {}).get("strategy_rates", {})).get("deflection", 0.0),
            "murderer_evasion_rate": summary.get("rq1", {}).get("rates_by_strategy", summary.get("rq1", {}).get("strategy_rates", {})).get("evasion_nonanswer", 0.0),
            "murderer_alibi_construction_rate": summary.get("rq1", {}).get("rates_by_strategy", {}).get("alibi_construction", 0.0),
            "murderer_accusation_redirection_rate": summary.get("rq1", {}).get("rates_by_strategy", {}).get("accusation_redirection", 0.0),
            "mean_accusation_confidence": summary.get("accusation_quality", {}).get("mean_confidence", 0.0),
            "structured_accusation_fraction": summary.get("accusation_quality", {}).get("structured_evidence_fraction", 0.0),
            "mean_accusation_evidence_item_count": summary.get("accusation_quality", {}).get("mean_evidence_item_count", 0.0),
            "murderer_vote_share": rq3.get("murderer_vote_share"),
            "belief_top1_alignment_fraction": rq3.get("belief_top1_alignment_fraction"),
            "belief_top3_alignment_fraction": rq3.get("belief_top3_alignment_fraction"),
            "mean_belief_uncertainty": rq3.get("mean_belief_uncertainty"),
            "belief_forced_top_suspect_count": rq3.get("belief_forced_top_suspect_count"),
            "group_solved": rq3.get("group_solved"),
            "murderer_escaped": rq3.get("murderer_escaped"),
            "random_vote_share_baseline": rq3.get("random_vote_share_baseline"),
            "random_group_solve_rate_baseline": rq3.get("random_group_solve_rate_baseline"),
            "random_escape_rate_baseline": rq3.get("random_escape_rate_baseline"),
            "run_usable_for_thesis": validation.get("run_usable_for_thesis"),
            "validation_status": validation.get("validation_status"),
            "validation_warning_count": len(validation.get("warnings", [])),
        })

    total_runs = len(aggregate_rows)
    deception_aggregate = aggregate_condition_deception(experiment_path)
    aggregate_summary = {
        "experiment_dir": str(experiment_path),
        "condition_name": summaries[0].get("condition_name"),
        "total_runs": total_runs,
        "thesis_usable_runs": usable_count,
        "thesis_usable_rate": usable_count / total_runs,
        "runs_with_quality_warnings": warning_count,
        "mean_total_turns": sum(row["total_turns"] for row in aggregate_rows) / total_runs,
        "mean_total_utterances": sum(row["total_utterances"] for row in aggregate_rows) / total_runs,
        "mean_total_belief_snapshots": sum((row.get("total_belief_snapshots") or 0) for row in aggregate_rows) / total_runs,
        "mean_murderer_speaker_share": sum(row["murderer_speaker_share"] for row in aggregate_rows) / total_runs,
        "mean_murderer_attention_received": sum(row["murderer_attention_received"] for row in aggregate_rows) / total_runs,
        "mean_murderer_followups_received": sum(row["murderer_followups_received"] for row in aggregate_rows) / total_runs,
        "mean_murderer_justification_requests_received": sum(row["murderer_justification_requests_received"] for row in aggregate_rows) / total_runs,
        "mean_murderer_pressure_signals_received": sum(row["murderer_pressure_signals_received"] for row in aggregate_rows) / total_runs,
        "mean_question_target_entropy": sum(row["question_target_entropy"] for row in aggregate_rows) / total_runs,
        "mean_pressure_target_gini": sum(row["pressure_target_gini"] for row in aggregate_rows) / total_runs,
        "mean_belief_top1_alignment_fraction": sum((row.get("belief_top1_alignment_fraction") or 0.0) for row in aggregate_rows) / total_runs,
        "mean_belief_top3_alignment_fraction": sum((row.get("belief_top3_alignment_fraction") or 0.0) for row in aggregate_rows) / total_runs,
        "mean_belief_uncertainty": sum((row.get("mean_belief_uncertainty") or 0.0) for row in aggregate_rows) / total_runs,
        "mean_belief_forced_top_suspect_count": sum((row.get("belief_forced_top_suspect_count") or 0.0) for row in aggregate_rows) / total_runs,
        "mean_murderer_labeled_utterances": sum((row["murderer_labeled_utterances"] or 0) for row in aggregate_rows) / total_runs,
        "mean_murderer_proportion_deceptive": sum((row.get("murderer_proportion_deceptive") or 0.0) for row in aggregate_rows) / total_runs,
        "mean_murderer_direct_denial_rate": sum((row["murderer_direct_denial_rate"] or 0.0) for row in aggregate_rows) / total_runs,
        "mean_murderer_deflection_rate": sum((row["murderer_deflection_rate"] or 0.0) for row in aggregate_rows) / total_runs,
        "mean_murderer_evasion_rate": sum((row["murderer_evasion_rate"] or 0.0) for row in aggregate_rows) / total_runs,
        "mean_murderer_alibi_construction_rate": sum((row.get("murderer_alibi_construction_rate") or 0.0) for row in aggregate_rows) / total_runs,
        "mean_murderer_accusation_redirection_rate": sum((row.get("murderer_accusation_redirection_rate") or 0.0) for row in aggregate_rows) / total_runs,
        "mean_accusation_confidence": sum(row["mean_accusation_confidence"] for row in aggregate_rows) / total_runs,
        "mean_structured_accusation_fraction": sum(row["structured_accusation_fraction"] for row in aggregate_rows) / total_runs,
        "mean_accusation_evidence_item_count": sum(row["mean_accusation_evidence_item_count"] for row in aggregate_rows) / total_runs,
        "mean_murderer_vote_share": sum((row["murderer_vote_share"] or 0.0) for row in aggregate_rows) / total_runs,
        "group_solve_rate": solve_count / total_runs,
        "murderer_escape_rate": escape_count / total_runs,
        "random_vote_share_baseline": sum((row["random_vote_share_baseline"] or 0.0) for row in aggregate_rows) / total_runs,
        "random_group_solve_rate_baseline": sum((row["random_group_solve_rate_baseline"] or 0.0) for row in aggregate_rows) / total_runs,
        "random_escape_rate_baseline": sum((row["random_escape_rate_baseline"] or 0.0) for row in aggregate_rows) / total_runs,
        "rq1": deception_aggregate,
    }

    _write_csv(experiment_path / "aggregate_runs.csv", aggregate_rows)
    with (experiment_path / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_summary, handle, indent=2, sort_keys=True)

    return aggregate_summary


def aggregate_experiment_conditions(experiment_dir: str | Path) -> Dict[str, Any]:
    experiment_path = Path(experiment_dir)
    experiment_path.mkdir(parents=True, exist_ok=True)
    conditions_dir = experiment_path / "conditions"

    if not conditions_dir.exists():
        summary = aggregate_experiment(experiment_path)
        with (experiment_path / "condition_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary

    condition_rows: List[Dict[str, Any]] = []
    for condition_path in sorted(path for path in conditions_dir.iterdir() if path.is_dir()):
        aggregate_path = condition_path / "aggregate_summary.json"
        if not aggregate_path.exists():
            aggregate_experiment(condition_path)
        if aggregate_path.exists():
            summary = _read_json(aggregate_path)
            summary["condition_name"] = summary.get("condition_name") or condition_path.name
            condition_rows.append(summary)

    if not condition_rows:
        return {"experiment_dir": str(experiment_path), "total_conditions": 0}

    _write_csv(
        experiment_path / "condition_summary.csv",
        [
            {
                "condition_name": row.get("condition_name"),
                "total_runs": row.get("total_runs"),
                "thesis_usable_runs": row.get("thesis_usable_runs"),
                "thesis_usable_rate": row.get("thesis_usable_rate"),
                "runs_with_quality_warnings": row.get("runs_with_quality_warnings"),
                "mean_total_turns": row.get("mean_total_turns"),
                "mean_total_utterances": row.get("mean_total_utterances"),
                "mean_murderer_speaker_share": row.get("mean_murderer_speaker_share"),
                "mean_murderer_attention_received": row.get("mean_murderer_attention_received"),
                "mean_murderer_followups_received": row.get("mean_murderer_followups_received"),
                "mean_murderer_justification_requests_received": row.get("mean_murderer_justification_requests_received"),
                "mean_murderer_pressure_signals_received": row.get("mean_murderer_pressure_signals_received"),
                "mean_question_target_entropy": row.get("mean_question_target_entropy"),
                "mean_pressure_target_gini": row.get("mean_pressure_target_gini"),
                "mean_murderer_labeled_utterances": row.get("mean_murderer_labeled_utterances"),
                "mean_murderer_direct_denial_rate": row.get("mean_murderer_direct_denial_rate"),
                "mean_murderer_deflection_rate": row.get("mean_murderer_deflection_rate"),
                "mean_murderer_evasion_rate": row.get("mean_murderer_evasion_rate"),
                "mean_accusation_confidence": row.get("mean_accusation_confidence"),
                "mean_structured_accusation_fraction": row.get("mean_structured_accusation_fraction"),
                "mean_accusation_evidence_item_count": row.get("mean_accusation_evidence_item_count"),
                "mean_murderer_vote_share": row.get("mean_murderer_vote_share"),
                "group_solve_rate": row.get("group_solve_rate"),
                "random_vote_share_baseline": row.get("random_vote_share_baseline"),
                "random_group_solve_rate_baseline": row.get("random_group_solve_rate_baseline"),
            }
            for row in condition_rows
        ],
    )

    summary = {
        "experiment_dir": str(experiment_path),
        "total_conditions": len(condition_rows),
        "conditions": condition_rows,
    }
    with (experiment_path / "condition_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
