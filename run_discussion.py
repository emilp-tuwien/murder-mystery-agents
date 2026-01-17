from typing import TypedDict, Dict, List, Optional, Literal
from pathlib import Path
from utils.agent_helper import load_character_descriptions
from graphs.discussion import build_graph
from schemas.state import GameState
from agents.agent import Agent
import sys
sys.path.insert(0, str(Path(__file__).parent / "game-master"))
from game_master import GameMaster

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import ollama
import json
load_dotenv()


def _banner(title: str, char: str = "=") -> None:
    width = max(60, len(title) + 12)
    line = char * width
    print("\n" + line)
    print(title.center(width))
    print(line)


def _section(title: str) -> None:
    width = max(50, len(title) + 10)
    line = "-" * width
    print("\n" + line)
    print(title.upper())
    print(line)


def _format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(no conversation)"

    rows = [f"IDX | TURN | SPEAKER             | TEXT",
            f"----+------+---------------------+--------------------------------------------------"]
    for idx, u in enumerate(history, start=1):
        turn = u.get("turn", idx)
        speaker = u.get("speaker", "Unknown")[:21]
        text = u.get("text", "").strip()
        rows.append(f"{idx:03d} | T{turn:02d} | {speaker:<21} | {text}")
    return "\n".join(rows)


def _get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models on the system."""
    try:
        models = ollama.list()
        # Extract model names from the response
        model_names = [model['name'] for model in models['models']]
        return sorted(model_names)
    except Exception as e:
        print(f"Warning: Could not fetch Ollama models: {e}")
        return []


def _select_ollama_model() -> Optional[str]:
    """Let user select from available Ollama models."""
    models = _get_available_ollama_models()
    
    if not models:
        print("No Ollama models found. Please install models using 'ollama pull <model_name>'")
        return None
    
    print("\nAvailable Ollama models:")
    for idx, model in enumerate(models, start=1):
        print(f"  {idx}. {model}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return None


if __name__ == "__main__":
    choice = input("Select LLM (g=GPT-4o-mini, o=Ollama): ").strip().lower()
    
    if choice == "g":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        print("Using GPT-4o-mini")
    elif choice == "o":
        selected_model = _select_ollama_model()
        if selected_model is None:
            print("No model selected. Exiting.")
            sys.exit(1)
        llm = ChatOllama(model=selected_model, temperature=0.7)
        print(f"Using {selected_model}")
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    roles_dir = Path(__file__).parent / "agents" / "roles"
    descriptions = load_character_descriptions(roles_dir)
    
    selected_characters = list(descriptions.keys())
    agents = {
        name: Agent(name, descriptions[name], llm)
        for name in selected_characters
    }
    print(f"Loaded agents: {list(agents.keys())}")

    # Initialize Game Master
    game_master = GameMaster(llm, list(agents.keys()))
    print("Game Master initialized.")

    app = build_graph(agents, game_master, max_turns=200)
    print("Discussion graph built (200 turns).")

    init: GameState = {
        "turn": 0,
        "history": [],
        "thoughts": {},
        "last_speaker": None,
        "pending_obligation": None,
        "next_speaker": None,
        "new_utterance": None,
        "done": False,
    }
    _banner("MURDER MYSTERY DISCUSSION")
    print("Starting discussion...\n")

    final = app.invoke(init, {"recursion_limit": 500})

    _banner("DISCUSSION COMPLETE - TIME TO VOTE")
    
    # Accusation phase
    _section("Accusation phase")
    print("Each player must now accuse someone of being the murderer.\n")
    
    agent_names = list(agents.keys())
    accusations = {}
    votes = {name: 0 for name in agent_names}
    
    for name, agent in agents.items():
        print(f"  {name} is deliberating...", end=" ")
        result = agent.accuse(final, agent_names)
        accusations[name] = result
        votes[result.accused] = votes.get(result.accused, 0) + 1
        print(f"accuses {result.accused}")
    
    _section("Final accusations")
    for name, result in accusations.items():
        print(f"\n{name} accuses {result.accused}:")
        print(f"  Reasoning: {result.reasoning}")
    
    _section("Vote tally")
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_votes:
        bar = "█" * count
        print(f"  {name}: {bar} ({count} votes)")
    
    # Determine winner
    max_votes = sorted_votes[0][1]
    winners = [name for name, count in sorted_votes if count == max_votes]
    
    _banner("GROUP VERDICT")
    if len(winners) == 1:
        print(f"🚨 THE GROUP HAS DECIDED: {winners[0]} IS THE MURDERER! 🚨")
    else:
        print(f"⚠️ TIE! The group suspects: {', '.join(winners)}")
    print("=" * 60)

    _section("Full transcript")
    print(_format_history(final["history"]))