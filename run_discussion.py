from typing import Optional
from pathlib import Path
from utils.agent_helper import load_character_descriptions, detect_murderer
from graphs.discussion import build_graph, visualize_graph
from schemas.state import GameState
from agents.agent import Agent
import sys
import os
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent / "game-master"))
from game_master import GameMaster
from utils.formatting import _banner, _section, _format_history
from utils.ollama_helper import _select_ollama_model

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

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
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nUsing default (20 conversations per round)")
            return 20


def _build_openai_llm(model_name: str, api_key: str, base_url: Optional[str] = None):
    kwargs = {
        "model": model_name,
        "temperature": 0.7,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _build_llm(choice: str):
    if choice == "l":
        normalized_base_url = _normalize_openai_base_url(LOCAL_LLM_API_URL)
        print(f"Using local hosted LLM via {normalized_base_url}")
        return _build_openai_llm(
            model_name=LOCAL_LLM_MODEL,
            api_key="not-needed",
            base_url=normalized_base_url,
        )

    if choice == "g":
        api_key = os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            print("\nNo OpenAI API key found in environment.")
            api_key = input("Paste your OpenAI API key: ").strip()
            if not api_key:
                print("No API key provided. Exiting.")
                sys.exit(1)
        else:
            use_existing = input(f"\nFound existing API key (ends with ...{api_key[-4:]}). Use it? (y/n): ").strip().lower()
            if use_existing != "y":
                api_key = input("Paste your OpenAI API key: ").strip()
                if not api_key:
                    print("No API key provided. Exiting.")
                    sys.exit(1)

        llm = _build_openai_llm(model_name="gpt-4o-mini", api_key=api_key)
        print("Using GPT-4o-mini")
        return llm

    if choice == "o":
        selected_model = _select_ollama_model()
        if selected_model is None:
            print("No model selected. Exiting.")
            sys.exit(1)
        llm = ChatOllama(model=selected_model, temperature=0.7)
        print(f"Using {selected_model}")
        return llm

    print("Invalid choice. Exiting.")
    sys.exit(1)


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


def run_game(model_choice: str, conversations_per_round: int = 20, enable_ui: bool = False, ui_port: int = 8000):
    llm = _build_llm(model_choice)

    max_turns = 6 * conversations_per_round + 10
    print(f"Game will have 6 rounds with {conversations_per_round} conversations per round.")
    print(f"Maximum turns: {max_turns}")

    ui_store = None
    if enable_ui:
        from ui.game_events import STORE
        from ui.server import start_ui_server
        ui_store = STORE
        ui_store.reset()
        try:
            start_ui_server(port=ui_port)
            print(f"Browser UI available at: http://127.0.0.1:{ui_port}")
        except PermissionError:
            print(f"UI server could not bind to port {ui_port} in this environment.")
            print("Continuing without live browser UI. Console output will still work.")
            ui_store = None

    roles_dir = Path(__file__).parent / "agents" / "roles"
    descriptions = load_character_descriptions(roles_dir)
    selected_characters = list(descriptions.keys())

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

    print(f"Loaded agents: {list(agents.keys())} ({len(agents)} agents)")
    if murderer_name:
        print(f"The murderer ({murderer_name}) knows they did it from Round 1.")

    game_master = GameMaster(llm, list(agents.keys()), conversations_per_round=conversations_per_round)
    print("Game Master initialized.")

    def emit_memory_snapshot():
        if ui_store is not None:
            ui_store.append("memory_updated", {
                "agent_memory": {
                    name: agent.export_memory_snapshot()
                    for name, agent in agents.items()
                }
            })

    app = build_graph(agents, game_master, max_turns=max_turns, ui_store=ui_store)
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
        "conversations_per_round": conversations_per_round,
        "history": [
            {
                "turn": 0,
                "speaker": "Game Master",
                "text": murder_announcement
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

    if ui_store is not None:
        ui_store.append("game_started", {
            "started_at": time.time(),
            "turn": 0,
            "round": 1,
            "phase": "introduction",
            "murderer": murderer_name,
        })
        ui_store.append("utterance", {"utterance": init["history"][0]})
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

        if ui_store is not None:
            ui_store.append("accusation", {
                "agent": name,
                "result": {
                    "accused": result.accused,
                    "reasoning": result.reasoning,
                }
            })

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

    if ui_store is not None:
        ui_store.append("game_finished", {
            "verdict": f"{verdict_text} {solved_text}",
            "murderer": murderer_name,
        })

    _section("Full transcript")
    print(_format_history(final["history"]))

    _section("Exporting agent thoughts to CSV")
    import csv
    from datetime import datetime

    thoughts_history = final.get("thoughts_history", [])
    if thoughts_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"agent_thoughts_{timestamp}.csv"

        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['turn', 'round', 'agent', 'action', 'importance', 'thought']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for record in thoughts_history:
                writer.writerow(record)

        print(f"Agent thoughts exported to: {csv_filename}")
        print(f"Total records: {len(thoughts_history)}")

        speak_count = sum(1 for r in thoughts_history if r['action'] == 'speak')
        listen_count = sum(1 for r in thoughts_history if r['action'] == 'listen')
        avg_importance = sum(r['importance'] for r in thoughts_history) / len(thoughts_history) if thoughts_history else 0
        print(f"Summary: {speak_count} speak decisions, {listen_count} listen decisions, avg importance: {avg_importance:.2f}")
    else:
        print("No thought records to export (Round 1 introductions don't have thoughts)")


def main():
    parser = argparse.ArgumentParser(description="Run the murder mystery discussion game.")
    parser.add_argument("--ui", action="store_true", help="Start the browser observer UI alongside the console output.")
    parser.add_argument("--ui-port", type=int, default=8000, help="Port for the local browser UI.")
    parser.add_argument("--model", choices=["prompt", "local", "gpt", "ollama"], default="prompt", help="Model backend to use.")
    parser.add_argument("--conversations-per-round", type=int, default=None, help="Number of conversations per round.")
    args = parser.parse_args()

    model_choice = _resolve_model_choice(args)
    conversations_per_round = _resolve_conversations_per_round(args)
    run_game(model_choice, conversations_per_round=conversations_per_round, enable_ui=args.ui, ui_port=args.ui_port)


if __name__ == "__main__":
    main()
