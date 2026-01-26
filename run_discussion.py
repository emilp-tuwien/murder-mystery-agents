from typing import TypedDict, Dict, List, Optional, Literal
from pathlib import Path
from utils.agent_helper import load_character_descriptions, load_round_description, load_confession, detect_murderer
from graphs.discussion import build_graph, visualize_graph
from schemas.state import GameState
from agents.agent import Agent
import sys
sys.path.insert(0, str(Path(__file__).parent / "game-master"))
from game_master import GameMaster
from utils.formatting import _banner, _section, _format_history
from utils.ollama_helper import _select_ollama_model, _get_available_ollama_models

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import subprocess
import json


load_dotenv()



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
        import os
        # Check if API key is already set in environment
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
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
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
    
    # Reset shared history singleton for new game
    from memory.agent_memory import SharedHistory
    SharedHistory.reset()
    
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
        "thoughts_history": [],  # Track all agent thoughts for CSV export
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
        print(f"THE GROUP HAS DECIDED: {winners[0]} IS THE MURDERER!")
    else:
        print(f"TIE! The group suspects: {', '.join(winners)}")
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
            print(f"CORRECT! {murderer_name} was indeed the murderer!")
            print("The group successfully solved the mystery!")
        else:
            print(f"WRONG! The real murderer was {murderer_name}!")
            print("The killer got away with it...")
    else:
        print("(Could not determine the actual murderer from the game files)")

    _section("Full transcript")
    print(_format_history(final["history"]))
    
    # Export agent thoughts to CSV
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
        
        # Print summary stats
        speak_count = sum(1 for r in thoughts_history if r['action'] == 'speak')
        listen_count = sum(1 for r in thoughts_history if r['action'] == 'listen')
        avg_importance = sum(r['importance'] for r in thoughts_history) / len(thoughts_history) if thoughts_history else 0
        print(f"Summary: {speak_count} speak decisions, {listen_count} listen decisions, avg importance: {avg_importance:.2f}")
    else:
        print("No thought records to export (Round 1 introductions don't have thoughts)")