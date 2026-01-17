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


if __name__ == "__main__":
    choice = input("Select LLM (g=GPT-4o-mini, l=Llama3.2): ").strip().lower()
    
    if choice == "g":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        print("Using GPT-4o-mini")
    else:
        llm = ChatOllama(model="llama3.2", temperature=0.7)
        print("Using Llama 3.2")

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

    app = build_graph(agents, game_master, max_turns=10)
    print("Discussion graph built (10 turns).")

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