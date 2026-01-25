#!/usr/bin/env python3
"""
Standalone script to visualize the Murder Mystery game graph.
Run this to generate a visualization without starting the full game.
"""

from pathlib import Path
from graphs.discussion import build_graph, visualize_graph
from schemas.state import GameState
import sys

# Mock classes for visualization (we don't need real LLM or agents)
class MockAgent:
    def __init__(self, name):
        self.name = name
    def think(self, state): pass
    def speak(self, state, constraint): return ""
    def update_round(self, round_num): pass

class MockGameMaster:
    def __init__(self, agent_names):
        self.agent_names = agent_names
        self.conversations_per_round = 20
        self.max_rounds = 6
    
    def decide_next_speaker(self, state, thoughts): 
        from game_master import SpeakerDecision
        return SpeakerDecision(
            reasoning="Mock", 
            next_speaker=self.agent_names[0],
            response_constraint=None,
            is_direct_address=False
        )
    
    def should_advance_round(self, convs, round_num): return convs >= self.conversations_per_round
    def get_phase_for_round(self, round_num): return "discussion" if round_num > 1 else "introduction"
    def is_game_complete(self, round_num, convs): return round_num >= 6
    def announce_round_change(self, new_round): return f"Round {new_round}"


def main():
    print("=" * 60)
    print("MURDER MYSTERY GAME GRAPH VISUALIZATION")
    print("=" * 60)
    
    # Create mock agents and game master
    agent_names = ["Agent A", "Agent B", "Agent C", "Agent D"]
    agents = {name: MockAgent(name) for name in agent_names}
    game_master = MockGameMaster(agent_names)
    
    # Build the graph
    print("\nBuilding graph...")
    app = build_graph(agents, game_master, max_turns=100)
    
    # Visualize
    print("\nGenerating visualization...")
    output_path = visualize_graph(app, "graphs/game_graph.png")
    
    if output_path:
        print(f"\n Graph visualization saved!")
        print(f"   Open: {output_path}")
    else:
        print("\n Failed to generate visualization")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent / "game-master"))
    sys.exit(main())
