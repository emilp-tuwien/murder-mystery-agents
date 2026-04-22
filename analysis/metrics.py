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


def _estimate_random_solve_rate(agent_names: List[str], murderer_name: str, trials: int = 5000) -> float:
    if not agent_names or murderer_name not in agent_names:
        return 0.0

    solved = 0
    for _ in range(trials):
        votes = Counter()
        for accuser in agent_names:
            candidates = [name for name in agent_names if name != accuser]
            accused = random.choice(candidates)
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

    for event in events:
        event_type = event.get("type")
        payload = event.get("payload", {})
        if event_type == "turn_started":
            current_round = payload.get("round", current_round)
            current_phase = payload.get("phase", current_phase)
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
        rows.append({
            "run_id": run_id,
            "accuser": agent,
            "accused": accused,
            "reasoning": result.get("reasoning", ""),
            "correct": accused == murderer_name,
            "accuser_is_murderer": agent == murderer_name,
            "accused_is_murderer": accused == murderer_name,
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


def _compute_rq3(accusations: List[Dict[str, Any]], agent_names: List[str], murderer_name: str) -> Dict[str, Any]:
    votes = Counter(row["accused"] for row in accusations if row.get("accused"))
    total_votes = len(accusations) or 1
    max_votes = max(votes.values()) if votes else 0
    winners = [name for name, count in votes.items() if count == max_votes] if max_votes else []
    murderer_vote_share = votes.get(murderer_name, 0) / total_votes
    return {
        "total_votes": len(accusations),
        "vote_counts": dict(votes),
        "murderer_vote_share": murderer_vote_share,
        "group_solved": murderer_name in winners,
        "winning_suspects": winners,
        "random_vote_share_baseline": 1 / len(agent_names) if agent_names else 0.0,
        "random_group_solve_rate_baseline": _estimate_random_solve_rate(agent_names, murderer_name),
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
    agent_rows, question_rows, mention_rows = _compute_agent_metrics(utterance_rows, agent_names, murderer_name)
    attention = build_attention_artifacts(run_path, utterance_rows, agent_names, murderer_name)
    rq3 = _compute_rq3(accusation_rows, agent_names, murderer_name)
    rq1 = label_deception_for_run(run_path, utterance_rows, manifest)

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
        "rq3": rq3,
    }

    _write_csv(run_path / "turns.csv", turn_rows)
    _write_csv(run_path / "utterances.csv", utterance_rows)
    _write_csv(run_path / "accusations.csv", accusation_rows)
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
    for summary in summaries:
        rq3 = summary.get("rq3", {})
        if rq3.get("group_solved"):
            solve_count += 1
        aggregate_rows.append({
            "run_id": summary.get("run_id"),
            "condition_name": summary.get("condition_name"),
            "murderer_name": summary.get("murderer_name"),
            "total_turns": summary.get("total_turns"),
            "total_utterances": summary.get("total_utterances"),
            "murderer_speaker_share": summary.get("rq2", {}).get("murderer_speaker_share"),
            "murderer_attention_received": summary.get("rq2", {}).get("murderer_attention_received"),
            "murderer_followups_received": summary.get("rq2", {}).get("murderer_followups_received"),
            "murderer_justification_requests_received": summary.get("rq2", {}).get("murderer_justification_requests_received"),
            "murderer_pressure_signals_received": summary.get("rq2", {}).get("murderer_pressure_signals_received"),
            "question_target_entropy": summary.get("rq2", {}).get("question_target_entropy"),
            "pressure_target_gini": summary.get("rq2", {}).get("pressure_target_gini"),
            "murderer_labeled_utterances": summary.get("rq1", {}).get("total_labeled_utterances"),
            "murderer_direct_denial_rate": summary.get("rq1", {}).get("strategy_rates", {}).get("direct_denial", 0.0),
            "murderer_deflection_rate": summary.get("rq1", {}).get("strategy_rates", {}).get("deflection", 0.0),
            "murderer_evasion_rate": summary.get("rq1", {}).get("strategy_rates", {}).get("evasion", 0.0),
            "murderer_vote_share": rq3.get("murderer_vote_share"),
            "group_solved": rq3.get("group_solved"),
            "random_vote_share_baseline": rq3.get("random_vote_share_baseline"),
            "random_group_solve_rate_baseline": rq3.get("random_group_solve_rate_baseline"),
        })

    total_runs = len(aggregate_rows)
    deception_aggregate = aggregate_condition_deception(experiment_path)
    aggregate_summary = {
        "experiment_dir": str(experiment_path),
        "condition_name": summaries[0].get("condition_name"),
        "total_runs": total_runs,
        "mean_total_turns": sum(row["total_turns"] for row in aggregate_rows) / total_runs,
        "mean_total_utterances": sum(row["total_utterances"] for row in aggregate_rows) / total_runs,
        "mean_murderer_speaker_share": sum(row["murderer_speaker_share"] for row in aggregate_rows) / total_runs,
        "mean_murderer_attention_received": sum(row["murderer_attention_received"] for row in aggregate_rows) / total_runs,
        "mean_murderer_followups_received": sum(row["murderer_followups_received"] for row in aggregate_rows) / total_runs,
        "mean_murderer_justification_requests_received": sum(row["murderer_justification_requests_received"] for row in aggregate_rows) / total_runs,
        "mean_murderer_pressure_signals_received": sum(row["murderer_pressure_signals_received"] for row in aggregate_rows) / total_runs,
        "mean_question_target_entropy": sum(row["question_target_entropy"] for row in aggregate_rows) / total_runs,
        "mean_pressure_target_gini": sum(row["pressure_target_gini"] for row in aggregate_rows) / total_runs,
        "mean_murderer_labeled_utterances": sum(row["murderer_labeled_utterances"] for row in aggregate_rows) / total_runs,
        "mean_murderer_direct_denial_rate": sum(row["murderer_direct_denial_rate"] for row in aggregate_rows) / total_runs,
        "mean_murderer_deflection_rate": sum(row["murderer_deflection_rate"] for row in aggregate_rows) / total_runs,
        "mean_murderer_evasion_rate": sum(row["murderer_evasion_rate"] for row in aggregate_rows) / total_runs,
        "mean_murderer_vote_share": sum(row["murderer_vote_share"] for row in aggregate_rows) / total_runs,
        "group_solve_rate": solve_count / total_runs,
        "random_vote_share_baseline": sum(row["random_vote_share_baseline"] for row in aggregate_rows) / total_runs,
        "random_group_solve_rate_baseline": sum(row["random_group_solve_rate_baseline"] for row in aggregate_rows) / total_runs,
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
