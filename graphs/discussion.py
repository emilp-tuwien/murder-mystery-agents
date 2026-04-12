from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
from schemas.state import GameState
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.judge import NormanResponseJudge
from utils.dialogue_analysis import detect_direct_address, extract_mentions, is_question


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _print_turn_header(turn: int, round_num: int, phase: str):
    """Print a clean turn header."""
    print(f"\n┌{'─'*68}┐")
    print(f"│  TURN {turn+1:<3} │ Round {round_num}/6 │ Phase: {phase.upper():<20}                │")
    print(f"└{'─'*68}┘")


def _print_speaker(speaker: str, text: str):
    """Print speaker's dialogue in a clean format."""
    print(f"\n {speaker}:")
    print(f"  ╭{'─'*64}╮")
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= 60:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    for line in lines:
        print(f"  │  {line:<62}│")
    print(f"  ╰{'─'*64}╯")


def _print_thinking_summary(thoughts: Dict, last_speaker: str = None):
    """Print a compact summary of agent thoughts."""
    if not thoughts:
        return
    print(f"\n  Agent Thoughts:")
    for name, tr in thoughts.items():
        action = " SPEAK" if tr.action == "speak" else " listen"
        urgency_bar = "█" * tr.importance + "░" * (9 - tr.importance)
        excluded = " (just spoke - excluded)" if name == last_speaker else ""
        print(f"     {name:<20} {action} [{urgency_bar}] {tr.importance}/9{excluded}")


def _print_gm_decision(speaker: str, reason: str = None):
    """Print game master's decision."""
    print(f"\n   Game Master selects: {speaker}")
    if reason:
        print(f"     └─ {reason}")


def _emit(ui_store, event_type: str, payload: Optional[Dict[str, Any]] = None):
    if ui_store is not None:
        ui_store.append(event_type, payload or {})


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH NODES
# ═══════════════════════════════════════════════════════════════════════════════

def think_all(state: GameState, agents: Dict[str, any], ui_store=None):
    current_round = state.get("current_round", 1)
    phase = state.get("phase", "introduction")
    turn = state.get("turn", 0)

    _print_turn_header(turn, current_round, phase)
    _emit(ui_store, "turn_started", {"turn": turn + 1, "round": current_round, "phase": phase})

    if current_round == 1:
        print(f"\n   Introduction round - agents will introduce themselves in order")
        intro_order = list(agents.keys())
        history = state.get("history", [])
        speakers_so_far = {u.get("speaker") for u in history}
        thoughts = {}
        thoughts_records = []

        for index, name in enumerate(intro_order):
            already_spoke = name in speakers_so_far
            importance = 9 if not already_spoke and index == len([n for n in intro_order if n in speakers_so_far]) else (2 if already_spoke else 6)
            action = "listen" if already_spoke else "speak"
            thought_text = (
                "It is my turn to introduce myself to the group."
                if not already_spoke
                else "I have already introduced myself and should let the others speak."
            )
            thoughts[name] = {
                "thought": thought_text,
                "action": action,
                "importance": importance,
                "reason_type": "introduction" if action == "speak" else "no_move",
            }
            thoughts_records.append({
                "turn": turn,
                "round": current_round,
                "agent": name,
                "action": action,
                "importance": importance,
                "reason_type": "introduction" if action == "speak" else "no_move",
                "thought": thought_text,
            })

        _emit(ui_store, "thoughts_generated", {"thoughts": thoughts})
        return {"thoughts": {}, "thoughts_history": thoughts_records}

    if not agents:
        print("  ERROR: No agents available!")
        _emit(ui_store, "game_error", {"error": "No agents available"})
        return {"thoughts": {}, "thoughts_history": []}

    print(f"\n  Agents are thinking...")

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
                print(f" Error in {name}'s thinking: {e}")
                from agents.agent import ThinkResult
                thoughts[name] = ThinkResult(thought="waiting", action="listen", importance=0, reason_type="no_move")

    pending = state.get("pending_obligation")
    must_respond = pending.get("addressee") if pending else None

    if must_respond and must_respond in thoughts:
        for name, tr in thoughts.items():
            if name == must_respond:
                tr.action = "speak"
                tr.importance = 9
                tr.reason_type = "direct_response"
                if "direct" not in tr.thought.lower() and "respond" not in tr.thought.lower():
                    tr.thought = f"I was directly addressed and need to respond now. {tr.thought}".strip()
            else:
                tr.action = "listen"
                tr.importance = min(tr.importance, 2)
                if getattr(tr, "reason_type", "no_move") == "direct_response":
                    tr.reason_type = "no_move"
    else:
        for name, tr in thoughts.items():
            if tr.importance >= 7:
                tr.action = "speak"
            elif tr.importance <= 2:
                tr.action = "listen"
                tr.reason_type = "no_move"
            else:
                tr.action = "listen"

    last_speaker = state.get("last_speaker")
    _print_thinking_summary(thoughts, last_speaker)
    _emit(ui_store, "thoughts_generated", {
        "thoughts": {
            name: {
                "thought": tr.thought,
                "action": tr.action,
                "importance": tr.importance,
                "reason_type": getattr(tr, "reason_type", "no_move"),
            }
            for name, tr in thoughts.items()
        }
    })

    thoughts_records = []
    for name, tr in thoughts.items():
        thoughts_records.append({
            "turn": turn,
            "round": current_round,
            "agent": name,
            "action": tr.action,
            "importance": tr.importance,
            "reason_type": getattr(tr, "reason_type", "no_move"),
            "thought": tr.thought
        })

    return {"thoughts": thoughts, "thoughts_history": thoughts_records}


def game_master_decide(state: GameState, game_master, agents: Dict[str, any], ui_store=None):
    """Game Master evaluates and decides who speaks next"""
    thoughts = state.get("thoughts", {})
    current_round = state.get("current_round", 1)

    if current_round == 1:
        agent_names = list(agents.keys())
        history = state.get("history", [])
        speakers_so_far = set(u["speaker"] for u in history)
        remaining = [name for name in agent_names if name not in speakers_so_far]

        if remaining:
            next_speaker = remaining[0]
            reason = f"Introduction ({len(remaining)} remaining)"
            _print_gm_decision(next_speaker, reason)
        else:
            next_speaker = agent_names[0]
            reason = "Fallback introduction order"

        _emit(ui_store, "speaker_selected", {"speaker": next_speaker, "reason": reason})
        return {"next_speaker": next_speaker, "pending_obligation": None}

    if not thoughts:
        print("  No thoughts available, skipping turn")
        _emit(ui_store, "speaker_selected", {"speaker": None, "reason": "No thoughts available"})
        return {"next_speaker": None, "pending_obligation": None}

    decision = game_master.decide_next_speaker(state, thoughts)
    _print_gm_decision(decision.next_speaker, decision.reasoning)

    if decision.is_direct_address:
        print(f"     ⚡ (Direct address - must respond)")

    _emit(ui_store, "speaker_selected", {
        "speaker": decision.next_speaker,
        "reason": decision.reasoning,
        "is_direct_address": decision.is_direct_address,
    })

    pending = None
    if decision.response_constraint:
        pending = {
            "addressee": decision.next_speaker,
            "response_constraint": decision.response_constraint,
            "from_speaker": state.get("last_speaker", ""),
            "from_text": state["history"][-1]["text"] if state.get("history") else "",
        }

    return {"next_speaker": decision.next_speaker, "pending_obligation": pending}


def speak(state: GameState, agents: Dict[str, any], ui_store=None, norman_judge=None):
    """Selected agent speaks"""
    speaker = state.get("next_speaker")

    if not speaker or speaker not in agents:
        print(f"   No speaker selected")
        return {"new_utterance": None, "last_speaker": state.get("last_speaker")}

    pending = state.get("pending_obligation")
    constraint = pending["response_constraint"] if pending and pending.get("addressee") == speaker else None

    text = agents[speaker].speak(state, response_constraint=constraint)
    agent_names = list(agents.keys())
    other_agents = [name for name in agent_names if name != speaker]
    addressed_to = detect_direct_address(text, other_agents)
    mentioned_agents = extract_mentions(text, other_agents)

    u = {
        "turn": state["turn"],
        "round": state.get("current_round"),
        "phase": state.get("phase"),
        "speaker": speaker,
        "text": text,
        "is_question": is_question(text),
        "addressed_to": addressed_to,
        "mentioned_agents": mentioned_agents,
        "response_to_speaker": pending.get("from_speaker") if pending and pending.get("addressee") == speaker else None,
    }
    _print_speaker(speaker, text)
    _emit(ui_store, "utterance", {"utterance": u})

    if speaker == "Norman D'Adly" and norman_judge is not None:
        recent_history = "\n".join([
            f"{item['speaker']}: {item['text']}" for item in state.get("history", [])[-5:]
        ])
        judgement = norman_judge.evaluate(recent_history=recent_history, norman_response=text)
        if judgement is not None:
            print(f"\n   Norman Judge: {judgement.verdict.upper()} - {judgement.reasoning}")
            print(f"     Follow-up: {judgement.follow_up_question}")
            _emit(ui_store, "norman_judged", {
                "verdict": judgement.verdict,
                "reasoning": judgement.reasoning,
                "follow_up_question": judgement.follow_up_question,
                "response": text,
            })

    return {"new_utterance": u, "last_speaker": speaker}


def update_history(state: GameState, agents: Dict[str, any]):
    """Update history and feed dialogue to agent memory systems."""
    u = state.get("new_utterance")
    if not u:
        return {"history": []}

    turn_id = u.get("turn", 0)
    speaker = u.get("speaker", "")
    text = u.get("text", "")

    for agent in agents.values():
        agent.memory.process_dialogue(turn_id, speaker, text)

    return {"history": [u]}


def check_round_advance(state: GameState, game_master, agents: Dict[str, any], ui_store=None):
    """Check if we should advance to the next round."""
    current_round = state.get("current_round", 1)
    conversations_in_round = state.get("conversations_in_round", 0) + 1
    history = state.get("history", [])

    if game_master.is_game_complete(current_round, conversations_in_round):
        print(f"\n{'═'*70}")
        print(f"  INVESTIGATION COMPLETE - Moving to accusation phase!")
        print(f"{'═'*70}\n")

        if history:
            print(f"   Summarizing Round {current_round} into bullet points...")
            bullets = game_master.summarize_round_history(history, current_round)
            for name, agent in agents.items():
                agent.add_round_summary(current_round, bullets)
            print(f"   Summary: {len(bullets)} key facts extracted")
            _emit(ui_store, "round_summarized", {"round": current_round, "bullets": bullets})

        _emit(ui_store, "round_changed", {"round": current_round, "phase": "accusation"})
        return {
            "conversations_in_round": conversations_in_round,
            "done": True,
            "phase": "accusation",
            "history": []
        }

    if current_round == 1:
        speakers_so_far = set(u["speaker"] for u in history)
        if state.get("new_utterance"):
            speakers_so_far.add(state["new_utterance"]["speaker"])

        if len(speakers_so_far) >= len(agents):
            new_round = 2
            new_phase = game_master.get_phase_for_round(new_round)

            print(f"\n   Summarizing Round {current_round} into bullet points...")
            bullets = game_master.summarize_round_history(history, current_round)
            for name, agent in agents.items():
                agent.add_round_summary(current_round, bullets)
            print(f"   Summary: {len(bullets)} key facts extracted")
            _emit(ui_store, "round_summarized", {"round": current_round, "bullets": bullets})

            announcement = game_master.announce_round_change(new_round)
            print(announcement)

            clue = game_master.get_clue_for_round(new_round)

            print(f"\n   Updating agent knowledge for Round {new_round}...")
            for name, agent in agents.items():
                agent.update_round(new_round)
                if clue:
                    agent.add_clue_to_memory(clue)
            print(f"   All agents updated with Round {new_round} information")
            _emit(ui_store, "round_changed", {"round": new_round, "phase": new_phase, "announcement": announcement})
            if clue:
                _emit(ui_store, "clue_revealed", {"round": new_round, "clue": clue})

            return {
                "current_round": new_round,
                "conversations_in_round": 0,
                "phase": new_phase,
                "history": []
            }

        return {"conversations_in_round": conversations_in_round}

    if game_master.should_advance_round(conversations_in_round, current_round):
        new_round = current_round + 1
        new_phase = game_master.get_phase_for_round(new_round)

        print(f"\n   Summarizing Round {current_round} into bullet points...")
        bullets = game_master.summarize_round_history(history, current_round)
        for name, agent in agents.items():
            agent.add_round_summary(current_round, bullets)
        print(f"   Summary: {len(bullets)} key facts extracted")
        _emit(ui_store, "round_summarized", {"round": current_round, "bullets": bullets})

        announcement = game_master.announce_round_change(new_round)
        print(announcement)

        clue = game_master.get_clue_for_round(new_round)

        print(f"\n   Updating agent knowledge for Round {new_round}...")
        for name, agent in agents.items():
            agent.update_round(new_round)
            if clue:
                agent.add_clue_to_memory(clue)
        print(f"   All agents updated with Round {new_round} information")
        _emit(ui_store, "round_changed", {"round": new_round, "phase": new_phase, "announcement": announcement})
        if clue:
            _emit(ui_store, "clue_revealed", {"round": new_round, "clue": clue})

        return {
            "current_round": new_round,
            "conversations_in_round": 0,
            "phase": new_phase,
            "history": []
        }

    return {"conversations_in_round": conversations_in_round}


def advance_turn(state: GameState, max_turns: int = 5):
    turn = state["turn"] + 1
    done = turn >= max_turns or state.get("done", False)
    return {"turn": turn, "done": done}


def route(state: GameState):
    if state.get("done"):
        print("  [ENDING] Discussion complete.")
        return END
    return "think_all"


def build_graph(agents: Dict[str, any], game_master, max_turns: int = 3, ui_store=None):

    norman_judge = NormanResponseJudge(game_master.llm)

    def route_fn(state: GameState):
        if state["turn"] >= max_turns or state.get("done", False):
            print(f"  [ENDING] Discussion complete at turn {state['turn']}.")
            return END
        return "think_all"

    g = StateGraph(GameState)

    g.add_node("think_all", lambda s: think_all(s, agents, ui_store=ui_store))
    g.add_node("game_master_decide", lambda s: game_master_decide(s, game_master, agents, ui_store=ui_store))
    g.add_node("speak", lambda s: speak(s, agents, ui_store=ui_store, norman_judge=norman_judge))
    g.add_node("update_history", lambda s: update_history(s, agents))
    g.add_node("check_round_advance", lambda s: check_round_advance(s, game_master, agents, ui_store=ui_store))
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
    from pathlib import Path

    try:
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "wb") as f:
            f.write(png_bytes)

        print(f"Graph visualization saved to: {output_file.absolute()}")
        return str(output_file.absolute())

    except Exception as e:
        print(f"Could not generate PNG visualization: {e}")
        print("Trying Mermaid text format instead...")

        try:
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
