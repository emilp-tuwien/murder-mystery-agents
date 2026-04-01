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
from schemas.state import GameState
from utils.agent_helper import detect_murderer, load_character_descriptions
from utils.formatting import _banner, _format_history, _section
from utils.ollama_helper import _select_ollama_model

sys.path.insert(0, str(Path(__file__).parent / "game-master"))
from game_master import GameMaster

load_dotenv()

LOCAL_LLM_API_URL = os.environ.get("LLM_API_URL", "http://192.168.11.11:8080/v1")
LOCAL_LLM_MODEL = os.environ.get("LLM_MODEL", "local_llm")


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


def _build_openai_llm(model_name: str, api_key: str, temperature: float, base_url: Optional[str] = None):
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
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
        )

    if config.backend == "gpt":
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"No API key found in environment variable {config.api_key_env}")
        model_name = config.model_name or "gpt-4o-mini"
        print(f"Using {model_name}")
        return _build_openai_llm(model_name=model_name, api_key=api_key, temperature=config.temperature)

    if config.backend == "ollama":
        model_name = config.model_name or _select_ollama_model()
        if model_name is None:
            raise RuntimeError("No Ollama model selected")
        print(f"Using {model_name}")
        return ChatOllama(model=model_name, temperature=config.temperature)

    raise RuntimeError(f"Unsupported backend: {config.backend}")


def _prompt_for_model_choice() -> str:
    print("Select LLM backend:")
    print("  l = Local hosted GPT-OSS (no auth)")
    print("  g = GPT-4o-mini (OpenAI API)")
    print("  o = Ollama")
    while True:
        choice = input("Choice (l/g/o): ").strip().lower()
        if choice in {"l", "g", "o"}:
            return choice
        print("Please enter l, g, or o.")


def _resolve_model_choice(args) -> str:
    if args.model == "local":
        return "l"
    if args.model == "gpt":
        return "g"
    if args.model == "ollama":
        return "o"
    return _prompt_for_model_choice()


def _resolve_conversations_per_round(args) -> int:
    if args.conversations_per_round is not None:
        return args.conversations_per_round
    return _select_conversations_per_round()


def _choice_to_backend(choice: str) -> str:
    mapping = {"l": "local", "g": "gpt", "o": "ollama"}
    return mapping[choice]


def run_game_from_config(config: RunConfig, event_sink=None) -> dict:
    llm = _build_llm_from_config(config)

    roles_dir = config.resolved_roles_dir()
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
        agents[name] = Agent(name, descriptions[name], llm, roles_dir, is_murderer=is_murderer)
        agents[name].update_round(1)

    if runtime_sink is not None:
        runtime_sink.append(
            "run_started",
            {
                "experiment_name": config.experiment_name,
                "replicate_id": config.replicate_id,
                "backend": config.backend,
                "model_name": config.model_name,
                "temperature": config.temperature,
                "conversations_per_round": config.conversations_per_round,
                "max_rounds": config.max_rounds,
                "agent_names": list(agents.keys()),
                "murderer_name": murderer_name,
                "scenario_id": config.scenario_id,
                "prompt_version": config.prompt_version,
                "turn_policy_version": config.turn_policy_version,
                "memory_version": config.memory_version,
            },
        )

    print(f"Loaded agents: {list(agents.keys())} ({len(agents)} agents)")
    if murderer_name:
        print(f"The murderer ({murderer_name}) knows they did it from Round 1.")

    game_master = GameMaster(
        llm,
        list(agents.keys()),
        conversations_per_round=config.conversations_per_round,
        max_rounds=config.max_rounds,
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

    murder_announcement = "ANNOUNCEMENT: Elizabeth Killingsworth has been found DEAD."
    init: GameState = {
        "turn": 0,
        "current_round": 1,
        "conversations_in_round": 0,
        "conversations_per_round": config.conversations_per_round,
        "history": [
            {
                "turn": 0,
                "round": 1,
                "phase": "introduction",
                "speaker": "Game Master",
                "text": murder_announcement,
                "is_question": False,
                "mentioned_agents": [],
            }
        ],
        "thoughts": {},
        "thoughts_history": [],
        "last_speaker": "Game Master",
        "pending_obligation": None,
        "next_speaker": None,
        "new_utterance": None,
        "done": False,
        "phase": "introduction",
    }

    if runtime_sink is not None:
        runtime_sink.append(
            "game_started",
            {
                "started_at": time.time(),
                "turn": 0,
                "round": 1,
                "phase": "introduction",
                "murderer": murderer_name,
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
        print(f"accuses {result.accused}")

        if runtime_sink is not None:
            runtime_sink.append(
                "accusation",
                {
                    "agent": name,
                    "result": {
                        "accused": result.accused,
                        "reasoning": result.reasoning,
                    },
                },
            )

    _section("Final accusations")
    for name, result in accusations.items():
        print(f"\n{name} accuses {result.accused}:")
        print(f"  Reasoning: {result.reasoning}")

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
        thoughts_csv = f"agent_thoughts_{timestamp}.csv"

        with open(thoughts_csv, "w", newline="") as csvfile:
            fieldnames = ["turn", "round", "agent", "action", "importance", "thought"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

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
    parser.add_argument("--model", choices=["prompt", "local", "gpt", "ollama"], default="prompt", help="Model backend to use.")
    parser.add_argument("--conversations-per-round", type=int, default=None, help="Number of conversations per round.")
    parser.add_argument("--max-rounds", type=int, default=6, help="Total rounds including accusation round.")
    args = parser.parse_args()

    model_choice = _resolve_model_choice(args)
    conversations_per_round = _resolve_conversations_per_round(args)
    config = RunConfig(
        backend=_choice_to_backend(model_choice),
        conversations_per_round=conversations_per_round,
        enable_ui=args.ui,
        ui_port=args.ui_port,
        max_rounds=args.max_rounds,
    )
    run_game_from_config(config)


if __name__ == "__main__":
    main()
