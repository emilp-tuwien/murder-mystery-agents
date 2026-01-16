from typing import Dict, List
from langgraph.graph import StateGraph, END
from schemas.state import GameState


def think_all(state: GameState, agents: Dict[str, any]):
    print(f"  [Turn {state['turn']}] History has {len(state.get('history', []))} messages. Agents thinking...")
    thoughts = {name: ag.think(state) for name, ag in agents.items()}
    for name, tr in thoughts.items():
        print(f"    {name}({'S' if tr.action == 'speak' else 'L'}:{tr.importance})")
    return {"thoughts": thoughts}


def game_master_decide(state: GameState, game_master, agents: Dict[str, any]):
    """Game Master evaluates and decides who speaks next"""
    thoughts = state.get("thoughts", {})
    
    if not thoughts:
        print("    [GM] No thoughts available, skipping")
        return {"next_speaker": None, "pending_obligation": None}
    
    decision = game_master.decide_next_speaker(state, thoughts)
    
    print(f"    [GM] Decision: {decision.next_speaker}")
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


def advance_turn(state: GameState, max_turns: int = 5):
    turn = state["turn"] + 1
    done = turn >= max_turns
    print(f"    → Turn {state['turn']} → {turn} (max: {max_turns}, done: {done})")
    
    # Show accumulated history
    if state.get("history"):
        print(f"    --- History so far ---")
        for u in state["history"]:
            print(f"      [{u['turn']}] {u['speaker']}: {u['text'][:60]}...")
        print(f"    ----------------------")
    
    return {"turn": turn, "done": done}


def route(state: GameState):
    if state.get("done"):
        print("  [ENDING] Discussion complete.")
        return END
    return "think_all"


def build_graph(agents: Dict[str, any], game_master, max_turns: int = 3):
    
    def route_fn(state: GameState):
        if state["turn"] >= max_turns:
            print(f"  [ENDING] Discussion complete at turn {state['turn']}.")
            return END
        return "think_all"
    
    g = StateGraph(GameState)

    g.add_node("think_all", lambda s: think_all(s, agents))
    g.add_node("game_master_decide", lambda s: game_master_decide(s, game_master, agents))
    g.add_node("speak", lambda s: speak(s, agents))
    g.add_node("update_history", update_history)
    g.add_node("advance_turn", lambda s: advance_turn(s, max_turns=max_turns))

    g.set_entry_point("think_all")
    g.add_edge("think_all", "game_master_decide")
    g.add_edge("game_master_decide", "speak")
    g.add_edge("speak", "update_history")
    g.add_edge("update_history", "advance_turn")
    g.add_conditional_edges("advance_turn", route_fn, {"think_all": "think_all", END: END})

    return g.compile()
