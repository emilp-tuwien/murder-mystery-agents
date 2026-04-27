from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import argparse
import csv
import os
import sys
import time

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from agents.agent import Agent
from experiments.config import RunConfig
from graphs.discussion import build_graph, visualize_graph
from instrumentation.event_logger import MultiEventSink
from scenarios import load_scenario_config
from schemas.state import GameState
from utils.agent_helper import detect_murderer, load_character_descriptions
from utils.evidence_gates import stage_name_for_round
from utils.formatting import _banner, _format_history, _section
from utils.ollama_helper import _select_ollama_model

sys.path.insert(0, str(Path(__file__).parent / "game-master"))
from game_master import GameMaster

load_dotenv()

LOCAL_LLM_API_URL = os.environ.get("LLM_API_URL", "http://192.168.11.11:8080/v1")
LOCAL_LLM_MODEL = os.environ.get("LLM_MODEL", "local_llm")
NVIDIA_API_URL = os.environ.get("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "moonshotai/kimi-k2.5")


def _normalize_openai_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    return url


def _select_conversations_per_round() -> int:
    while True:
        try:
            convs = input("\nConversations per round? (default: 20): ").strip()
            if not convs:
                return 20
            num_convs = int(convs)
            if num_convs > 0:
                return num_convs
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nUsing default (20 conversations per round)")
            return 20


def _build_openai_llm(
    model_name: str,
    api_key: str,
    temperature: float,
    base_url: Optional[str] = None,
    seed: Optional[int] = None,
    extra_model_kwargs: Optional[dict[str, Any]] = None,
):
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if seed is not None:
        kwargs["seed"] = seed
    if extra_model_kwargs:
        kwargs["model_kwargs"] = extra_model_kwargs
    return ChatOpenAI(**kwargs)


def _build_llm_from_config(config: RunConfig) -> Any:
    if config.backend == "local":
        base_url = _normalize_openai_base_url(config.base_url or LOCAL_LLM_API_URL)
        model_name = config.model_name or LOCAL_LLM_MODEL
        print(f"Using local hosted LLM via {base_url}")
        return _build_openai_llm(
            model_name=model_name,
            api_key="not-needed",
            base_url=base_url,
            temperature=config.temperature,
            seed=config.resolved_seed(),
        )

    if config.backend == "gpt":
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"No API key found in environment variable {config.api_key_env}")
        model_name = config.model_name or "gpt-4o-mini"
        print(f"Using {model_name}")
        return _build_openai_llm(model_name=model_name, api_key=api_key, temperature=config.temperature, seed=config.resolved_seed())

    if config.backend == "nvidia":
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"No API key found in environment variable {config.api_key_env}")
        base_url = _normalize_openai_base_url(config.base_url or NVIDIA_API_URL)
        model_name = config.model_name or NVIDIA_MODEL
        print(f"Using NVIDIA-hosted model {model_name} via {base_url}")
        extra_model_kwargs = None
        if config.enable_thinking:
            extra_model_kwargs = {"chat_template_kwargs": {"thinking": True}}
        return _build_openai_llm(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=config.temperature,
            seed=config.resolved_seed(),
            extra_model_kwargs=extra_model_kwargs,
        )

    if config.backend == "ollama":
        model_name = config.model_name or _select_ollama_model()
        if model_name is None:
            raise RuntimeError("No Ollama model selected")
        print(f"Using {model_name}")
        kwargs = {"model": model_name, "temperature": config.temperature}
        if config.resolved_seed() is not None:
            kwargs["seed"] = config.resolved_seed()
        return ChatOllama(**kwargs)

    raise RuntimeError(f"Unsupported backend: {config.backend}")


def _prompt_for_model_choice() -> str:
    print("Select LLM backend:")
    print("  l = Local hosted GPT-OSS (no auth)")
    print("  g = GPT-4o-mini (OpenAI API)")
    print("  n = NVIDIA-hosted model (OpenAI-compatible API)")
    print("  o = Ollama")
    while True:
        choice = input("Choice (l/g/n/o): ").strip().lower()
        if choice in {"l", "g", "n", "o"}:
            return choice
        print("Please enter l, g, n, or o.")


def _resolve_model_choice(args) -> str:
    if args.model == "local":
        return "l"
    if args.model == "gpt":
        _ensure_openai_api_key("OPENAI_API_KEY")
        return "g"
    if args.model == "nvidia":
        _ensure_openai_api_key("NVIDIA_API_KEY")
        return "n"
    if args.model == "ollama":
        return "o"
    choice = _prompt_for_model_choice()
    if choice == "g":
        _ensure_openai_api_key("OPENAI_API_KEY")
    if choice == "n":
        _ensure_openai_api_key("NVIDIA_API_KEY")
    return choice


def _resolve_conversations_per_round(args) -> int:
    if args.conversations_per_round is not None:
        return args.conversations_per_round
    return _select_conversations_per_round()


def _choice_to_backend(choice: str) -> str:
    mapping = {"l": "local", "g": "gpt", "n": "nvidia", "o": "ollama"}
    return mapping[choice]


def _ensure_openai_api_key(api_key_env: str = "OPENAI_API_KEY") -> None:
    existing = os.environ.get(api_key_env, "").strip()
    if existing:
        return

    print(f"No {api_key_env} found.")
    try:
        api_key = input(f"Please enter your OpenAI API key for {api_key_env}: ").strip()
    except KeyboardInterrupt:
        print("\nOpenAI API key entry cancelled.")
        raise RuntimeError(f"No API key provided for {api_key_env}")

    if not api_key:
        raise RuntimeError(f"No API key provided for {api_key_env}")

    os.environ[api_key_env] = api_key


def run_game_from_config(config: RunConfig, event_sink=None) -> dict:
    llm = _build_llm_from_config(config)

    roles_dir = config.resolved_roles_dir()
    clues_dir = config.resolved_clues_dir()
    scenario = load_scenario_config(config.resolved_scenario_path())
    descriptions = load_character_descriptions(roles_dir)
    selected_characters = list(descriptions.keys())
    max_turns = len(selected_characters) + max(config.max_rounds - 2, 0) * config.conversations_per_round + 10

    print(f"Game will have {config.max_rounds} rounds with {config.conversations_per_round} conversations per round.")
    print(f"Maximum turns: {max_turns}")

    ui_store = None
    runtime_sink = event_sink
    if config.enable_ui:
        from ui.game_events import STORE
        from ui.server import start_ui_server

        ui_store = STORE
        ui_store.reset()
        runtime_sink = MultiEventSink([event_sink, ui_store])
        try:
            start_ui_server(port=config.ui_port)
            print(f"Browser UI available at: http://127.0.0.1:{config.ui_port}")
        except PermissionError:
            print(f"UI server could not bind to port {config.ui_port} in this environment.")
            print("Continuing without live browser UI. Console output will still work.")
            runtime_sink = event_sink
            ui_store = None

    from memory.agent_memory import SharedHistory

    SharedHistory.reset()

    agents = {}
    murderer_name = None
    for name in selected_characters:
        is_murderer = detect_murderer(roles_dir, name)
        if is_murderer:
            murderer_name = name
            print(f"  [Detected murderer: {name}]")
        agents[name] = Agent(name, descriptions[name], llm, roles_dir, is_murderer=is_murderer, scenario=scenario)
        agents[name].update_round(1)

    if runtime_sink is not None:
        runtime_sink.append(
            "run_started",
            {
                "experiment_name": config.experiment_name,
                "replicate_id": config.replicate_id,
                "seed": config.resolved_seed(),
                "backend": config.backend,
                "model_name": config.model_name,
                "temperature": config.temperature,
                "conversations_per_round": config.conversations_per_round,
                "max_rounds": config.max_rounds,
                "stage_gate_policy": config.stage_gate_policy,
                "min_round_gate_conversations": config.resolved_min_round_gate_conversations(),
                "max_round_gate_conversations": config.resolved_max_round_gate_conversations(),
                "agent_names": list(agents.keys()),
                "murderer_name": murderer_name,
                "scenario_id": config.scenario_id,
                "scenario_title": scenario.title,
                "scenario_location": scenario.location,
                "prompt_version": config.prompt_version,
                "turn_policy_version": config.turn_policy_version,
                "memory_version": config.memory_version,
            },
        )

    print(f"Loaded agents: {list(agents.keys())} ({len(agents)} agents)")
    if murderer_name:
        print(f"The murderer ({murderer_name}) knows they did it from Round 1.")
        print(f"Scenario loaded: {scenario.title} ({scenario.scenario_id})")

    game_master = GameMaster(
        llm,
        list(agents.keys()),
        conversations_per_round=config.conversations_per_round,
        max_rounds=config.max_rounds,
        clues_dir=clues_dir,
        scenario=scenario,
        stage_gate_policy=config.stage_gate_policy,
        min_round_gate_conversations=config.resolved_min_round_gate_conversations(),
        max_round_gate_conversations=config.resolved_max_round_gate_conversations(),
        min_unique_question_targets_per_round=config.min_unique_question_targets_per_round,
        min_question_coverage_fraction_per_round=config.min_question_coverage_fraction_per_round,
        min_evidence_signals_per_round=config.min_evidence_signals_per_round,
        min_pressure_signals_per_round=config.min_pressure_signals_per_round,
        min_clue_references_per_round=config.min_clue_references_per_round,
        min_synthesis_signals_final_round=config.min_synthesis_signals_final_round,
    )
    print("Game Master initialized.")

    def emit_memory_snapshot():
        if runtime_sink is not None:
            runtime_sink.append(
                "memory_updated",
                {
                    "agent_memory": {
                        name: agent.export_memory_snapshot()
                        for name, agent in agents.items()
                    }
                },
            )

    app = build_graph(agents, game_master, max_turns=max_turns, ui_store=runtime_sink)
    print("Discussion graph built.")

    print("Generating graph visualization...")
    visualize_graph(app, "graphs/game_graph.png")

    initial_context = game_master.provide_initial_context()
    print(initial_context)

    murder_announcement = f"ANNOUNCEMENT: {scenario.victim_status_line}"
    init: GameState = {
        "turn": 0,
        "current_round": 1,
        "current_stage": stage_name_for_round(1, config.max_rounds),
        "conversations_in_round": 0,
        "conversations_per_round": config.conversations_per_round,
        "history": [
            {
                "turn": 0,
                "round": 1,
                "phase": "introduction",
                "stage": stage_name_for_round(1, config.max_rounds),
                "speaker": "Game Master",
                "text": murder_announcement,
                "is_question": False,
                "mentioned_agents": [],
            }
        ],
        "thoughts": {},
        "thoughts_history": [],
        "belief_snapshots": {},
        "belief_history": [],
        "last_speaker": "Game Master",
        "pending_obligation": None,
        "next_speaker": None,
        "new_utterance": None,
        "done": False,
        "phase": "introduction",
        "round_gate_status": {},
    }

    if runtime_sink is not None:
        runtime_sink.append(
            "game_started",
            {
                "started_at": time.time(),
                "turn": 0,
                "round": 1,
                "phase": "introduction",
                "stage": stage_name_for_round(1, config.max_rounds),
                "murderer": murderer_name,
                "scenario_title": scenario.title,
                "scenario_location": scenario.location,
            },
        )
        runtime_sink.append("utterance", {"utterance": init["history"][0]})
        emit_memory_snapshot()

    _banner("MURDER MYSTERY DISCUSSION")
    print("Starting Round 1: Introductions...\n")

    final = app.invoke(init, {"recursion_limit": 1000})
    emit_memory_snapshot()

    _banner("DISCUSSION COMPLETE - TIME TO VOTE")

    _section("Accusation phase")
    print("Each player must now accuse someone of being the murderer.")
    print("Remember: Players CANNOT accuse themselves!\n")

    agent_names = list(agents.keys())
    accusations = {}
    votes = {name: 0 for name in agent_names}

    for name, agent in agents.items():
        print(f"  {name} is deliberating...", end=" ")
        result = agent.accuse(final, agent_names)

        if result.accused == name:
            other_agents = [n for n in agent_names if n != name]
            print(f"(tried to accuse self, redirecting)...", end=" ")
            result.accused = other_agents[0] if other_agents else result.accused

        accusations[name] = result
        votes[result.accused] = votes.get(result.accused, 0) + 1
        accusation_context = getattr(agent, "last_accusation_context", {}) or {}
        print(f"accuses {result.accused}")

        if runtime_sink is not None:
            runtime_sink.append(
                "accusation",
                {
                    "agent": name,
                    "result": {
                        "accused": result.accused,
                        "reasoning": result.reasoning,
                        "confidence": result.confidence,
                        "primary_basis": result.primary_basis,
                        "evidence_items": result.evidence_items,
                        "motive_case": result.motive_case,
                        "means_case": result.means_case,
                        "opportunity_case": result.opportunity_case,
                        "contradiction_case": result.contradiction_case,
                        "comparative_case": result.comparative_case,
                        "uncertainty": result.uncertainty,
                    },
                    "belief_snapshot": accusation_context.get("belief_snapshot", {}),
                    "belief_alignment": {
                        "top_n_candidates": accusation_context.get("top_n_candidates", []),
                        "top_suspect": accusation_context.get("top_suspect"),
                        "accused_rank": accusation_context.get("accused_rank"),
                        "accused_in_top_n": accusation_context.get("accused_in_top_n"),
                        "corrected_to_top_suspect": accusation_context.get("corrected_to_top_suspect", False),
                    },
                },
            )

    _section("Final accusations")
    for name, result in accusations.items():
        accusation_context = getattr(agents.get(name), "last_accusation_context", {}) or {}
        print(f"\n{name} accuses {result.accused}:")
        print(f"  Confidence: {result.confidence}/100")
        if accusation_context.get("top_suspect"):
            print(f"  Belief state: top suspect was {accusation_context.get('top_suspect')} | rank of accused: {accusation_context.get('accused_rank')} | in top-{len(accusation_context.get('top_n_candidates', [])) or 0}: {accusation_context.get('accused_in_top_n')}")
        print(f"  Primary basis: {result.primary_basis}")
        print(f"  Reasoning: {result.reasoning}")
        if result.evidence_items:
            print("  Evidence:")
            for item in result.evidence_items:
                print(f"    - {item}")
        if result.uncertainty:
            print(f"  Uncertainty: {result.uncertainty}")

    _section("Vote tally")
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_votes:
        bar = "█" * count
        print(f"  {name}: {bar} ({count} votes)")

    max_votes = sorted_votes[0][1]
    winners = [name for name, count in sorted_votes if count == max_votes]

    _banner("GROUP VERDICT")
    if len(winners) == 1:
        verdict_text = f"THE GROUP HAS DECIDED: {winners[0]} IS THE MURDERER!"
        print(verdict_text)
    else:
        verdict_text = f"TIE! The group suspects: {', '.join(winners)}"
        print(verdict_text)
    print("=" * 60)

    _banner("CONFESSION TIME - THE TRUTH REVEALED")
    print("\nEach player now reveals their secrets...\n")

    for name, agent in agents.items():
        _section(f"{name}'s Confession")
        confession = agent.load_confession()
        if confession:
            print(confession)
        else:
            print("(No confession available)")
        print()

    _banner("FINAL VERDICT")
    if murderer_name:
        if murderer_name in winners:
            print(f"CORRECT! {murderer_name} was indeed the murderer!")
            print("The group successfully solved the mystery!")
            solved_text = f"Solved correctly. Real murderer: {murderer_name}."
        else:
            print(f"WRONG! The real murderer was {murderer_name}!")
            print("The killer got away with it...")
            solved_text = f"Wrong verdict. Real murderer: {murderer_name}."
    else:
        solved_text = "Could not determine the actual murderer from the game files."
        print("(Could not determine the actual murderer from the game files)")

    if runtime_sink is not None:
        runtime_sink.append(
            "game_finished",
            {
                "verdict": f"{verdict_text} {solved_text}",
                "murderer": murderer_name,
                "winners": winners,
                "votes": votes,
            },
        )

    _section("Full transcript")
    print(_format_history(final["history"]))

    _section("Exporting agent thoughts to CSV")
    from datetime import datetime

    thoughts_history = final.get("thoughts_history", [])
    thoughts_csv = None
    if thoughts_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interactive_dir = config.resolved_output_root() / "interactive"
        interactive_dir.mkdir(parents=True, exist_ok=True)
        thoughts_csv = str(interactive_dir / f"agent_thoughts_{timestamp}.csv")

        with open(thoughts_csv, "w", newline="") as csvfile:
            fieldnames = ["turn", "round", "agent", "action", "importance", "reason_type", "thought"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")

            writer.writeheader()
            for record in thoughts_history:
                writer.writerow(record)

        print(f"Agent thoughts exported to: {thoughts_csv}")
        print(f"Total records: {len(thoughts_history)}")

        speak_count = sum(1 for r in thoughts_history if r["action"] == "speak")
        listen_count = sum(1 for r in thoughts_history if r["action"] == "listen")
        avg_importance = sum(r["importance"] for r in thoughts_history) / len(thoughts_history) if thoughts_history else 0
        print(f"Summary: {speak_count} speak decisions, {listen_count} listen decisions, avg importance: {avg_importance:.2f}")
    else:
        print("No thought records to export (Round 1 introductions don't have thoughts)")

    return {
        "murderer_name": murderer_name,
        "agent_names": agent_names,
        "votes": votes,
        "winners": winners,
        "thoughts_csv": thoughts_csv,
        "group_solved": murderer_name in winners if murderer_name else False,
    }


def run_game(model_choice: str, conversations_per_round: int = 20, enable_ui: bool = False, ui_port: int = 8000):
    config = RunConfig(
        backend=_choice_to_backend(model_choice),
        conversations_per_round=conversations_per_round,
        enable_ui=enable_ui,
        ui_port=ui_port,
    )
    return run_game_from_config(config)


def main():
    parser = argparse.ArgumentParser(description="Run the murder mystery discussion game.")
    parser.add_argument("--ui", action="store_true", help="Start the browser observer UI alongside the console output.")
    parser.add_argument("--ui-port", type=int, default=8000, help="Port for the local browser UI.")
    parser.add_argument("--model", choices=["prompt", "local", "gpt", "nvidia", "ollama"], default="prompt", help="Model backend to use.")
    parser.add_argument("--model-name", default=None, help="Override the model name for the selected backend.")
    parser.add_argument("--base-url", default=None, help="Override the API base URL for OpenAI-compatible backends.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for the selected backend.")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable provider-specific thinking/reasoning mode when supported (e.g. NVIDIA Kimi via chat_template_kwargs).")
    parser.add_argument("--conversations-per-round", type=int, default=None, help="Number of conversations per round.")
    parser.add_argument("--max-rounds", type=int, default=6, help="Total rounds including accusation round.")
    parser.add_argument("--scenario-id", default="business-of-murder-v1", help="Scenario identifier for logging/display.")
    parser.add_argument("--scenario-path", default=None, help="Path to scenario.json for alternate scenarios.")
    parser.add_argument("--roles-dir", default=None, help="Override roles directory.")
    parser.add_argument("--clues-dir", default=None, help="Override clues directory.")
    args = parser.parse_args()

    model_choice = _resolve_model_choice(args)
    conversations_per_round = _resolve_conversations_per_round(args)
    backend = _choice_to_backend(model_choice)
    api_key_env = "OPENAI_API_KEY" if backend == "gpt" else "NVIDIA_API_KEY" if backend == "nvidia" else "OPENAI_API_KEY"

    config = RunConfig(
        backend=backend,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key_env=api_key_env,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        conversations_per_round=conversations_per_round,
        enable_ui=args.ui,
        ui_port=args.ui_port,
        max_rounds=args.max_rounds,
        scenario_id=args.scenario_id,
        scenario_path=args.scenario_path,
        roles_dir=args.roles_dir,
        clues_dir=args.clues_dir,
    )
    run_game_from_config(config)


if __name__ == "__main__":
    main()
