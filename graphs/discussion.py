from typing import Dict, List
from langgraph.graph import StateGraph, END
from schemas.state import GameState
from concurrent.futures import ThreadPoolExecutor, as_completed


def think_all(state: GameState, agents: Dict[str, any]):
    current_round = state.get("current_round", 1)
    phase = state.get("phase", "introduction")
    
    # In round 1 (introduction), skip thinking - just cycle through agents
    if current_round == 1:
        print(f"  [Turn {state['turn']}] Round 1 (introductions) - No thinking needed")
        # Return empty thoughts - game master will handle round 1 differently
        return {"thoughts": {}}
    
    print(f"  [Turn {state['turn']}] Round {current_round} ({phase}) - History: {len(state.get('history', []))} msgs. Agents thinking...")
    
    if not agents:
        print("    ERROR: No agents available!")
        return {"thoughts": {}}
    
    # Parallelize agent thinking for speed
    thoughts = {}
    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as executor:
        future_to_agent = {
            executor.submit(ag.think, state): name 
            for name, ag in agents.items()
        }
        for future in as_completed(future_to_agent):
            name = future_to_agent[future]
            try:
                thoughts[name] = future.result()
            except Exception as e:
                print(f"    Error in {name}'s thinking: {e}")
                # Fallback
                from agents.agent import ThinkResult
                thoughts[name] = ThinkResult(thought="waiting", action="listen", importance=1)
    
    for name, tr in thoughts.items():
        print(f"    {name}({'S' if tr.action == 'speak' else 'L'}:{tr.importance})")
    return {"thoughts": thoughts}


def game_master_decide(state: GameState, game_master, agents: Dict[str, any]):
    """Game Master evaluates and decides who speaks next"""
    thoughts = state.get("thoughts", {})
    current_round = state.get("current_round", 1)
    phase = state.get("phase", "introduction")
    
    # Round 1: Each agent introduces themselves exactly once
    if current_round == 1:
        agent_names = list(agents.keys())
        history = state.get("history", [])
        
        # Find who has already introduced themselves
        speakers_so_far = set(u["speaker"] for u in history)
        
        # Find agents who haven't spoken yet
        remaining = [name for name in agent_names if name not in speakers_so_far]
        
        if remaining:
            next_speaker = remaining[0]
            print(f"    [GM] Round 1 (introductions) - {next_speaker} will introduce themselves ({len(remaining)} remaining)")
        else:
            # Everyone has introduced themselves - this shouldn't happen as round should advance
            next_speaker = agent_names[0]
            print(f"    [GM] Round 1 complete - all agents have introduced themselves")
        
        return {"next_speaker": next_speaker, "pending_obligation": None}
    
    # Round 2+: Use thinking-based decision
    if not thoughts:
        print("    [GM] No thoughts available, skipping")
        return {"next_speaker": None, "pending_obligation": None}
    
    decision = game_master.decide_next_speaker(state, thoughts)
    
    print(f"    [GM] Round {current_round} ({phase}) - Decision: {decision.next_speaker}")
    print(f"         Reason: {decision.reasoning}")
    if decision.is_direct_address:
        print(f"         (Direct address - must respond)")
    
    pending = None
    if decision.response_constraint:
        pending = {
            "addressee": decision.next_speaker,
            "response_constraint": decision.response_constraint,
            "from_speaker": state.get("last_speaker", ""),
            "from_text": state["history"][-1]["text"] if state.get("history") else "",
        }
    
    return {"next_speaker": decision.next_speaker, "pending_obligation": pending}


def speak(state: GameState, agents: Dict[str, any]):
    """Selected agent speaks"""
    speaker = state.get("next_speaker")
    
    if not speaker or speaker not in agents:
        print(f"    → No speaker selected")
        return {"new_utterance": None, "last_speaker": state.get("last_speaker")}
    
    pending = state.get("pending_obligation")
    constraint = pending["response_constraint"] if pending and pending.get("addressee") == speaker else None
    
    text = agents[speaker].speak(state, response_constraint=constraint)

    u = {"turn": state["turn"], "speaker": speaker, "text": text}
    print(f"    → {speaker}: {text}")
    return {"new_utterance": u, "last_speaker": speaker}


def update_history(state: GameState):
    u = state.get("new_utterance")
    if not u:
        print(f"    [No new utterance to add]")
        return {"history": []}  # Empty list - nothing to add
    
    print(f"    [Adding to history: {u['speaker']}: {u['text'][:50]}...]")
    # Return list with single item - reducer will append it
    return {"history": [u]}


def check_round_advance(state: GameState, game_master, agents: Dict[str, any]):
    """Check if we should advance to the next round."""
    current_round = state.get("current_round", 1)
    conversations_in_round = state.get("conversations_in_round", 0) + 1  # +1 for the conversation just added
    
    # Check if game is complete (after round 5)
    if game_master.is_game_complete(current_round, conversations_in_round):
        print(f"\n{'='*60}")
        print(f"  GAME COMPLETE - Moving to accusation phase!")
        print(f"{'='*60}\n")
        return {
            "conversations_in_round": conversations_in_round,
            "done": True,
            "phase": "accusation"
        }
    
    # Round 1 special case: End after everyone has introduced themselves
    if current_round == 1:
        history = state.get("history", [])
        # Count unique speakers (add 1 for the utterance just added)
        speakers_so_far = set(u["speaker"] for u in history)
        if state.get("new_utterance"):
            speakers_so_far.add(state["new_utterance"]["speaker"])
        
        # If all agents have spoken, advance to round 2
        if len(speakers_so_far) >= len(agents):
            new_round = 2
            new_phase = game_master.get_phase_for_round(new_round)
            
            print(game_master.announce_round_change(new_round))
            
            # Update all agents with new round information
            for name, agent in agents.items():
                agent.update_round(new_round)
                print(f"    [Updated {name} with Round {new_round} information]")
            
            return {
                "current_round": new_round,
                "conversations_in_round": 0,
                "phase": new_phase
            }
        
        return {"conversations_in_round": conversations_in_round}
    
    # Round 2+: Check if we should advance based on conversation count
    if game_master.should_advance_round(conversations_in_round, current_round):
        new_round = current_round + 1
        new_phase = game_master.get_phase_for_round(new_round)
        
        print(game_master.announce_round_change(new_round))
        
        # Update all agents with new round information
        for name, agent in agents.items():
            agent.update_round(new_round)
            print(f"    [Updated {name} with Round {new_round} information]")
        
        return {
            "current_round": new_round,
            "conversations_in_round": 0,
            "phase": new_phase
        }
    
    return {"conversations_in_round": conversations_in_round}


def advance_turn(state: GameState, max_turns: int = 5):
    turn = state["turn"] + 1
    current_round = state.get("current_round", 1)
    conversations_in_round = state.get("conversations_in_round", 0)
    done = turn >= max_turns or state.get("done", False)
    
    print(f"    → Turn {state['turn']} → {turn} | Round {current_round} | Conv in round: {conversations_in_round} | Done: {done}")
    
    return {"turn": turn, "done": done}


def route(state: GameState):
    if state.get("done"):
        print("  [ENDING] Discussion complete.")
        return END
    return "think_all"


def build_graph(agents: Dict[str, any], game_master, max_turns: int = 3):
    
    def route_fn(state: GameState):
        if state["turn"] >= max_turns or state.get("done", False):
            print(f"  [ENDING] Discussion complete at turn {state['turn']}.")
            return END
        return "think_all"
    
    g = StateGraph(GameState)

    g.add_node("think_all", lambda s: think_all(s, agents))
    g.add_node("game_master_decide", lambda s: game_master_decide(s, game_master, agents))
    g.add_node("speak", lambda s: speak(s, agents))
    g.add_node("update_history", update_history)
    g.add_node("check_round_advance", lambda s: check_round_advance(s, game_master, agents))
    g.add_node("advance_turn", lambda s: advance_turn(s, max_turns=max_turns))

    g.set_entry_point("think_all")
    g.add_edge("think_all", "game_master_decide")
    g.add_edge("game_master_decide", "speak")
    g.add_edge("speak", "update_history")
    g.add_edge("update_history", "check_round_advance")
    g.add_edge("check_round_advance", "advance_turn")
    g.add_conditional_edges("advance_turn", route_fn, {"think_all": "think_all", END: END})

    return g.compile()


def visualize_graph(compiled_graph, output_path: str = "graphs/game_graph.png"):
    """
    Visualize the LangGraph and save it as an image.
    
    Args:
        compiled_graph: The compiled LangGraph
        output_path: Path to save the visualization (PNG)
    """
    from pathlib import Path
    
    try:
        # Get the graph visualization as PNG bytes
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the PNG
        with open(output_file, "wb") as f:
            f.write(png_bytes)
        
        print(f"Graph visualization saved to: {output_file.absolute()}")
        return str(output_file.absolute())
    
    except Exception as e:
        print(f"Could not generate PNG visualization: {e}")
        print("Trying Mermaid text format instead...")
        
        try:
            # Fallback: Generate mermaid diagram text
            mermaid_code = compiled_graph.get_graph().draw_mermaid()
            
            output_file = Path(output_path).with_suffix(".md")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w") as f:
                f.write("# Murder Mystery Game Graph\n\n")
                f.write("```mermaid\n")
                f.write(mermaid_code)
                f.write("\n```\n")
            
            print(f"Mermaid diagram saved to: {output_file.absolute()}")
            print("\nMermaid code:\n")
            print(mermaid_code)
            return str(output_file.absolute())
        
        except Exception as e2:
            print(f"Could not generate mermaid visualization: {e2}")
            return None
