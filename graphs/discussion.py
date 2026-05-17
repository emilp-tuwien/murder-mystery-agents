from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
from schemas.state import GameState
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.judge import MurdererResponseJudge
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
    current_stage = state.get("current_stage", phase)
    turn = state.get("turn", 0)

    _print_turn_header(turn, current_round, phase)
    _emit(ui_store, "turn_started", {"turn": turn + 1, "round": current_round, "phase": phase, "stage": current_stage})

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
        speakers_so_far = set(u["speaker"] for u in history if u.get("speaker") != "Game Master")
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


def speak(state: GameState, agents: Dict[str, any], ui_store=None, murderer_judge=None, murderer_name: str | None = None):
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
        "stage": state.get("current_stage"),
        "speaker": speaker,
        "text": text,
        "is_question": is_question(text),
        "addressed_to": addressed_to,
        "mentioned_agents": mentioned_agents,
        "response_to_speaker": pending.get("from_speaker") if pending and pending.get("addressee") == speaker else None,
    }
    _print_speaker(speaker, text)
    _emit(ui_store, "utterance", {"utterance": u})

    if murderer_name and speaker == murderer_name and murderer_judge is not None:
        recent_history = "\n".join([
            f"{item['speaker']}: {item['text']}" for item in state.get("history", [])[-5:]
        ])
        judgement = murderer_judge.evaluate(recent_history=recent_history, murderer_response=text)
        if judgement is not None:
            print(f"\n   Murderer Judge: {judgement.verdict.upper()} - {judgement.reasoning}")
            print(f"     Follow-up: {judgement.follow_up_question}")
            _emit(ui_store, "murderer_judged", {
                "suspect": speaker,
                "verdict": judgement.verdict,
                "reasoning": judgement.reasoning,
                "follow_up_question": judgement.follow_up_question,
                "response": text,
            })

    return {"new_utterance": u, "last_speaker": speaker}


def update_history(state: GameState, agents: Dict[str, any], ui_store=None):
    """Update history, feed dialogue to memory systems, and log belief-state updates."""
    u = state.get("new_utterance")
    if not u:
        return {"history": []}

    turn_id = u.get("turn", 0)
    speaker = u.get("speaker", "")
    text = u.get("text", "")
    agent_names = list(agents.keys())

    belief_snapshots = {}
    belief_history = []
    first = True
    for name, agent in agents.items():
        agent.memory.process_dialogue(turn_id, speaker, text, update_shared=first)
        first = False
        snapshot = agent.observe_utterance(u, agent_names)
        belief_snapshots[name] = snapshot
        belief_history.append(snapshot)

    _emit(ui_store, "beliefs_updated", {
        "turn": turn_id,
        "round": u.get("round"),
        "stage": u.get("stage"),
        "observed_speaker": speaker,
        "beliefs": belief_snapshots,
    })

    # Clear any pending direct-response obligation now that the addressee has
    # actually spoken — otherwise the next think_all will keep boosting them to
    # importance=9 / reason=direct_response and the GM's continuation fallback
    # will re-pick the same speaker turn after turn.
    pending = state.get("pending_obligation")
    obligation_satisfied = bool(pending and pending.get("addressee") == speaker)

    update = {"history": [u], "belief_snapshots": belief_snapshots, "belief_history": belief_history}
    if obligation_satisfied:
        update["pending_obligation"] = None
    return update


def check_round_advance(state: GameState, game_master, agents: Dict[str, any], ui_store=None):
    """Check if we should advance to the next round/stage."""
    current_round = state.get("current_round", 1)
    conversations_in_round = state.get("conversations_in_round", 0) + 1
    history = state.get("history", [])
    current_stage = state.get("current_stage", game_master.get_stage_for_round(current_round))
    # Slice to only this round's utterances so summarizer and gate don't see prior rounds.
    current_round_history = [u for u in history if u.get("round") == current_round]

    def _summarize_current_round():
        if current_round_history:
            print(f"\n   Summarizing Round {current_round} into bullet points...")
            bullets = game_master.summarize_round_history(current_round_history, current_round)
            for _, agent in agents.items():
                agent.add_round_summary(current_round, bullets)
            print(f"   Summary: {len(bullets)} key facts extracted")
            _emit(ui_store, "round_summarized", {"round": current_round, "stage": current_stage, "bullets": bullets})

    def _advance_to_round(new_round: int, advance_reason: str, gate_payload: Optional[Dict[str, Any]] = None):
        new_phase = game_master.get_phase_for_round(new_round)
        new_stage = game_master.get_stage_for_round(new_round)
        decision_payload = {
            "from_round": current_round,
            "from_stage": current_stage,
            "to_round": new_round,
            "to_stage": new_stage,
            "phase": new_phase,
            "advance_reason": advance_reason,
            "conversations_in_round": conversations_in_round,
            "conversations_per_round": state.get("conversations_per_round"),
        }
        if gate_payload:
            decision_payload.update(gate_payload)
        _emit(ui_store, "round_advance_decision", decision_payload)

        _summarize_current_round()

        announcement = game_master.announce_round_change(new_round)
        print(announcement)

        clue = game_master.get_clue_for_round(new_round)

        if clue:
            # Inject the clue into SharedHistory as a Game Master message so every agent
            # sees it in their [CONVERSATION] section on the next think/speak call.
            from memory.agent_memory import SharedHistory
            SharedHistory().append(state.get("turn", 0), "Game Master", f"[NEW EVIDENCE] {clue}")

        print(f"\n   Updating agent knowledge for Round {new_round}...")
        for _, agent in agents.items():
            agent.update_round(new_round)
            if clue:
                agent.add_clue_to_memory(clue)
        print(f"   All agents updated with Round {new_round} information")
        _emit(ui_store, "round_changed", {"round": new_round, "phase": new_phase, "stage": new_stage, "announcement": announcement})
        if clue:
            _emit(ui_store, "clue_revealed", {"round": new_round, "stage": new_stage, "clue": clue})

        return {
            "current_round": new_round,
            "current_stage": new_stage,
            "conversations_in_round": 0,
            "phase": new_phase,
            "round_gate_status": gate_payload or {},
        }

    if game_master.is_game_complete(current_round, conversations_in_round):
        _emit(ui_store, "round_advance_decision", {
            "from_round": current_round,
            "from_stage": current_stage,
            "to_round": current_round,
            "to_stage": "accusation",
            "phase": "accusation",
            "advance_reason": "investigation_complete",
            "conversations_in_round": conversations_in_round,
        })
        print(f"\n{'═'*70}")
        print("  INVESTIGATION COMPLETE - Moving to accusation phase!")
        print(f"{'═'*70}\n")

        _summarize_current_round()

        _emit(ui_store, "round_changed", {"round": current_round, "phase": "accusation", "stage": "accusation"})
        return {
            "conversations_in_round": conversations_in_round,
            "done": True,
            "phase": "accusation",
            "current_stage": "accusation",
        }

    if current_round == 1:
        speakers_so_far = set(u["speaker"] for u in history if u.get("speaker") != "Game Master")

        if len(speakers_so_far) >= len(agents):
            return _advance_to_round(
                2,
                "all_introductions_completed",
                {
                    "speakers_so_far": len(speakers_so_far),
                    "total_agents": len(agents),
                },
            )

        return {"conversations_in_round": conversations_in_round, "current_stage": current_stage}

    assessment = game_master.should_advance_round(current_round_history, conversations_in_round, current_round)
    gate_payload = {
        "stage_gate_policy": assessment.gate_policy,
        "stage_name": assessment.stage_name,
        "gate_satisfied": assessment.gate_satisfied,
        "allow_advance": assessment.allow_advance,
        "advance_reason": assessment.advance_reason,
        "minimum_conversations_reached": assessment.minimum_conversations_reached,
        "hard_cap_reached": assessment.hard_cap_reached,
        "unmet_requirements": assessment.unmet_requirements,
        "metrics": assessment.metrics,
        "thresholds": assessment.thresholds,
        "clue_available": assessment.clue_available,
        "clue_keywords": assessment.clue_keywords,
    }

    if assessment.allow_advance:
        return _advance_to_round(current_round + 1, assessment.advance_reason, gate_payload)

    return {
        "conversations_in_round": conversations_in_round,
        "current_stage": current_stage,
        "round_gate_status": gate_payload,
    }

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

    murderer_name = next((name for name, agent in agents.items() if getattr(agent, "is_murderer", False)), None)
    murderer_judge = MurdererResponseJudge(game_master.llm, murderer_name=murderer_name or "the suspect")

    def route_fn(state: GameState):
        if state["turn"] >= max_turns or state.get("done", False):
            print(f"  [ENDING] Discussion complete at turn {state['turn']}.")
            return END
        return "think_all"

    g = StateGraph(GameState)

    g.add_node("think_all", lambda s: think_all(s, agents, ui_store=ui_store))
    g.add_node("game_master_decide", lambda s: game_master_decide(s, game_master, agents, ui_store=ui_store))
    g.add_node("speak", lambda s: speak(s, agents, ui_store=ui_store, murderer_judge=murderer_judge, murderer_name=murderer_name))
    g.add_node("update_history", lambda s: update_history(s, agents, ui_store=ui_store))
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
