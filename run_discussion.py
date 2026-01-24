from typing import TypedDict, Dict, List, Optional, Literal
from pathlib import Path
from utils.agent_helper import load_character_descriptions, load_round_description, load_confession, detect_murderer
from graphs.discussion import build_graph, visualize_graph
from schemas.state import GameState
from agents.agent import Agent
import sys
sys.path.insert(0, str(Path(__file__).parent / "game-master"))
from game_master import GameMaster

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import subprocess
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
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output - skip header line and extract model names
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            return []
        
        models = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if parts:
                # Model name is the first column
                model_name = parts[0]
                models.append(model_name)
        
        return sorted(models)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not fetch Ollama models: {e}")
        return []
    except FileNotFoundError:
        print("Warning: 'ollama' command not found. Please install Ollama.")
        return []
    except Exception as e:
        print(f"Warning: Error fetching Ollama models: {e}")
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


def _select_conversations_per_round() -> int:
    """Let user specify the number of conversations per round."""
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

    conversations_per_round = _select_conversations_per_round()
    # Total turns = 5 rounds (intro + 4 discussion) * conversations per round + buffer
    max_turns = 6 * conversations_per_round + 10  # 6 rounds with buffer
    print(f"Game will have 6 rounds with {conversations_per_round} conversations per round.")
    print(f"Maximum turns: {max_turns}")

    roles_dir = Path(__file__).parent / "agents" / "roles"
    descriptions = load_character_descriptions(roles_dir)
    
    selected_characters = list(descriptions.keys())
    
    # Detect murderer and create agents
    agents = {}
    murderer_name = None
    for name in selected_characters:
        is_murderer = detect_murderer(roles_dir, name)
        if is_murderer:
            murderer_name = name
            print(f"  [Detected murderer: {name}]")
        agents[name] = Agent(name, descriptions[name], llm, roles_dir, is_murderer=is_murderer)
        # Initialize agents with round 1 information
        agents[name].update_round(1)
    
    print(f"Loaded agents: {list(agents.keys())} ({len(agents)} agents)")
    if murderer_name:
        print(f"The murderer ({murderer_name}) knows they did it from Round 1.")

    # Initialize Game Master with conversations per round setting
    game_master = GameMaster(llm, list(agents.keys()), conversations_per_round=conversations_per_round)
    print("Game Master initialized.")

    app = build_graph(agents, game_master, max_turns=max_turns)
    print(f"Discussion graph built.")
    
    # Visualize the graph
    print("Generating graph visualization...")
    visualize_graph(app, "graphs/game_graph.png")

    # Display initial game context
    initial_context = game_master.provide_initial_context()
    print(initial_context)

    # Create initial context message that will be in history for all agents
    murder_announcement = """ANNOUNCEMENT: Elizabeth Killingsworth has been found DEAD.""" 
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
        "last_speaker": "Game Master",
        "pending_obligation": None,
        "next_speaker": None,
        "new_utterance": None,
        "done": False,
        "phase": "introduction",
    }
    _banner("MURDER MYSTERY DISCUSSION")
    print("Starting Round 1: Introductions...\n")

    final = app.invoke(init, {"recursion_limit": 1000})

    _banner("DISCUSSION COMPLETE - TIME TO VOTE")
    
    # Accusation phase
    _section("Accusation phase")
    print("Each player must now accuse someone of being the murderer.")
    print("Remember: Players CANNOT accuse themselves!\n")
    
    agent_names = list(agents.keys())
    accusations = {}
    votes = {name: 0 for name in agent_names}
    
    for name, agent in agents.items():
        print(f"  {name} is deliberating...", end=" ")
        result = agent.accuse(final, agent_names)
        
        # Double-check: prevent self-accusation
        if result.accused == name:
            other_agents = [n for n in agent_names if n != name]
            print(f"(tried to accuse self, redirecting)...", end=" ")
            result.accused = other_agents[0] if other_agents else result.accused
        
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
    
    # CONFESSION PHASE - Everyone reads their confession
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
    
    # Reveal if the group got it right
    _banner("FINAL VERDICT")
    if murderer_name:
        if murderer_name in winners:
            print(f"✅ CORRECT! {murderer_name} was indeed the murderer!")
            print("The group successfully solved the mystery!")
        else:
            print(f"❌ WRONG! The real murderer was {murderer_name}!")
            print("The killer got away with it...")
    else:
        print("(Could not determine the actual murderer from the game files)")

    _section("Full transcript")
    print(_format_history(final["history"]))