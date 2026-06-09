from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
import PyPDF2

from scenarios import ScenarioConfig
from utils.dialogue_analysis import detect_direct_address, detect_direct_address_llm
from utils.evidence_gates import RoundGateAssessment, assess_round_gate, stage_name_for_round
from utils.formatting import _boxed


class SpeakerDecision(BaseModel):
    """Game Master's decision on who should speak next"""
    reasoning: str = Field(description="Brief reasoning for the decision")
    next_speaker: str = Field(description="Name of the player who should speak next")
    response_constraint: Optional[str] = Field(default=None, description="What they should respond to, if applicable")
    is_direct_address: bool = Field(default=False, description="True if someone was directly asked/addressed")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        if isinstance(data, dict):
            if "next_speaker" not in data and "decision" in data:
                data = dict(data)
                data["next_speaker"] = data["decision"]
        return data

    @field_validator("next_speaker", mode="before")
    @classmethod
    def normalize_next_speaker(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


class RoundSummary(BaseModel):
    """Bullet point summary of a round's key events"""
    bullets: List[str] = Field(description="List of key facts revealed in this round")


class GameMaster:
    def __init__(
        self,
        llm: Any,
        agent_names: List[str],
        conversations_per_round: int = 20,
        max_rounds: int = 6,
        clues_dir: Optional[Path] = None,
        scenario: Optional[ScenarioConfig] = None,
        stage_gate_policy: str = "round_budget",
        min_round_gate_conversations: int = 6,
        max_round_gate_conversations: Optional[int] = None,
        min_unique_question_targets_per_round: int = 3,
        min_question_coverage_fraction_per_round: float = 0.50,
        min_evidence_signals_per_round: int = 3,
        min_pressure_signals_per_round: int = 2,
        min_clue_references_per_round: int = 1,
        min_synthesis_signals_final_round: int = 1,
    ):
        self.llm = llm
        self.agent_names = agent_names
        self.llm_decide = llm.with_structured_output(SpeakerDecision, method="json_mode")
        self.persona = self._load_persona()
        self.conversations_per_round = conversations_per_round
        self.max_rounds = max_rounds
        self.clues_dir = clues_dir or (Path(__file__).parent.parent / "clues")
        self.scenario = scenario or ScenarioConfig()
        self.stage_gate_policy = stage_gate_policy
        self.min_round_gate_conversations = min_round_gate_conversations
        self.max_round_gate_conversations = max_round_gate_conversations or max(conversations_per_round, min_round_gate_conversations + 6)
        self.min_unique_question_targets_per_round = min_unique_question_targets_per_round
        self.min_question_coverage_fraction_per_round = min_question_coverage_fraction_per_round
        self.min_evidence_signals_per_round = min_evidence_signals_per_round
        self.min_pressure_signals_per_round = min_pressure_signals_per_round
        self.min_clue_references_per_round = min_clue_references_per_round
        self.min_synthesis_signals_final_round = min_synthesis_signals_final_round
    
    def _load_persona(self) -> str:
        """Load game master description from PDF"""
        pdf_path = Path(__file__).parent / "description" / "game-master.pdf"
        if pdf_path.exists():
            try:
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    return text.strip() if text.strip() else self._default_persona()
            except Exception as e:
                print(f"Warning: Could not load game master PDF: {e}")
                return self._default_persona()
        return self._default_persona()
    
    def _default_persona(self) -> str:
        return """You are the Game Master of a murder mystery party.
Your role is to facilitate the discussion and ensure the investigation progresses.
You decide who speaks next based on the conversation flow."""

    def _load_clue(self, clue_number: int) -> str:
        """Load a clue from the clues folder."""
        clue_path = self.clues_dir / f"clue{clue_number}.txt"
        if clue_path.exists():
            try:
                return clue_path.read_text().strip()
            except Exception as e:
                print(f"Warning: Could not load clue {clue_number}: {e}")
                return ""
        return ""

    def summarize_round_history(self, history: List[dict], round_num: int) -> List[str]:
        """
        Summarize a round's conversation into bullet points.
        Called at the end of each round to compress history.
        """
        if not history:
            return []
        
        # Format history for the LLM
        history_txt = "\n".join([
            f"{msg['speaker']}: {msg['text']}" for msg in history
        ])
        
        llm_summary = self.llm.with_structured_output(RoundSummary, method="json_mode")
        
        msgs = [
            SystemMessage(content="""You are summarizing a murder mystery discussion round.
Extract ONLY the key facts, revelations, accusations, and alibis mentioned.
Each bullet should be one specific fact (who said what, what was revealed).
Be concise - max 10 words per bullet. No opinions, just facts.
Return valid JSON with key: bullets."""),
            HumanMessage(content=f"""Round {round_num} conversation:
{history_txt}

Create bullet points of key facts revealed (max 15 bullets):"""),
        ]
        
        try:
            result = llm_summary.invoke(msgs)
            return result.bullets if result else []
        except Exception as e:
            print(f"Warning: Could not summarize round {round_num}: {e}")
            # Fallback: extract speaker statements as simple bullets
            bullets = []
            for msg in history[-10:]:  # Last 10 messages as fallback
                bullets.append(f"{msg['speaker']}: {msg['text'][:50]}...")
            return bullets

    def assess_round_progress(
        self,
        history: List[dict],
        current_round: int,
        conversations_in_round: int,
        repetition_tracker=None,
    ) -> RoundGateAssessment:
        current_clue = self.get_clue_for_round(current_round)
        return assess_round_gate(
            history=history,
            agent_names=self.agent_names,
            current_round=current_round,
            conversations_in_round=conversations_in_round,
            max_rounds=self.max_rounds,
            gate_policy=self.stage_gate_policy,
            clue_text=current_clue,
            min_conversations=self.min_round_gate_conversations,
            hard_cap_conversations=self.max_round_gate_conversations,
            min_unique_question_targets=self.min_unique_question_targets_per_round,
            min_question_coverage_fraction=self.min_question_coverage_fraction_per_round,
            min_evidence_signals=self.min_evidence_signals_per_round,
            min_pressure_signals=self.min_pressure_signals_per_round,
            min_clue_references=self.min_clue_references_per_round,
            min_synthesis_signals=self.min_synthesis_signals_final_round,
            stopwords=getattr(self.scenario, "gate_stopwords", None),
            evidence_patterns=getattr(self.scenario, "gate_evidence_patterns", None),
            pressure_patterns=getattr(self.scenario, "gate_pressure_patterns", None),
            synthesis_patterns=getattr(self.scenario, "gate_synthesis_patterns", None),
            repetition_tracker=repetition_tracker,
        )

    def should_advance_round(
        self,
        history: List[dict],
        conversations_in_round: int,
        current_round: int,
        repetition_tracker=None,
    ) -> RoundGateAssessment:
        """Assess whether the current investigation round should advance."""
        return self.assess_round_progress(history, current_round, conversations_in_round, repetition_tracker)
    
    def get_stage_for_round(self, round_num: int) -> str:
        return stage_name_for_round(round_num, self.max_rounds)

    def get_phase_for_round(self, round_num: int) -> str:
        """
        Determine the game phase based on current round.
        Round 1: introduction
        Rounds 2..max_rounds-1: discussion
        Round max_rounds: accusation
        """
        if round_num == 1:
            return "introduction"
        if round_num < self.max_rounds:
            return "discussion"
        return "accusation"
    
    def is_game_complete(self, current_round: int, conversations_in_round: int) -> bool:
        """Check whether the investigation is over and the accusation phase begins.

        The FINAL discussion round is ``max_rounds - 1``: the last clues (#4 fire
        escape and #5 timeline — the only two that actually incriminate the murderer)
        are revealed when entering it and debated there. Round ``max_rounds`` is the
        accusation phase itself and runs NO discussion turns, so the game is complete
        once we reach it. (Completion is driven directly by ``check_round_advance`` via
        ``_complete_investigation``; this method is kept for external callers/tests.)
        """
        return current_round >= self.max_rounds

    def get_clue_for_round(self, new_round: int) -> str:
        """Return the clue revealed when entering a new round.

        The authored role descriptions gate clues as: no clue in Rounds 1-2,
        Clue #1 in Round 3, Clue #2 in Round 4, Clue #3 in Round 5, and the
        remaining clues (#4 and #5) at the final accusation stage. So the clue
        number trails the round by two; entering Round 3 reveals Clue #1.
        Keeping this aligned prevents agents from "knowing" crime-scene details
        (e.g. the keychain/blood smear in Clue #1) a round before their own
        briefing mentions them.
        """
        clue_number = new_round - 2
        if clue_number < 1:
            return ""
        return self._load_clue(clue_number)

    def _clue_count(self) -> int:
        """Number of clue files available for this scenario."""
        count = 0
        while (self.clues_dir / f"clue{count + 1}.txt").exists():
            count += 1
        return count

    def get_remaining_clues(self, current_round: int) -> List[str]:
        """Return every not-yet-revealed clue, in order, for the final stage.

        The per-round schedule (see ``get_clue_for_round``) reveals clues up to
        number ``current_round - 2`` by the time the investigation completes at
        ``current_round``. This returns the rest — clues
        ``current_round - 1 .. clue_count`` — so the accusation phase delivers
        Clue #4 and Clue #5 together, matching the Round 6 briefings.
        """
        first_unrevealed = max(1, current_round - 1)
        clues = []
        for clue_number in range(first_unrevealed, self._clue_count() + 1):
            text = self._load_clue(clue_number)
            if text:
                clues.append(text)
        return clues

    def decide_next_speaker(self, state: dict, thoughts: dict, repetition_tracker=None) -> SpeakerDecision:
        """
        Evaluate the last message and all agent thoughts to decide who speaks next.
        
        Priority:
        1. If someone was directly addressed/asked a question → they MUST respond
        2. Otherwise, pick the agent with highest urgency score (excluding last speaker)
        """
        history = state.get("history", [])
        last_utterance = history[-1] if history else None
        current_round = state.get("current_round", 1)
        phase = state.get("phase", "introduction")
        
        # Available speakers (exclude last speaker to avoid monopolization)
        last_speaker = state.get("last_speaker")
        available = [n for n in self.agent_names if n != last_speaker] if last_speaker else self.agent_names

        # TOPIC-SATURATION CIRCUIT BREAKER (runs BEFORE the direct-address mandate).
        # A rotating interrogation loop — different askers pressing different people
        # about the same subject, turn after turn — is invisible to the per-question
        # nagging check (the asker/addressee pair keeps changing). Left alone, the
        # unconditional direct-address mandate sustains it indefinitely. When the
        # recent window is both topically clustered AND no longer producing facts,
        # we DEMOTE the mandate and hand the floor to a quiet agent with an explicit
        # instruction to open a new line of inquiry.
        #
        # EXCEPTION: a fresh direct question that OPENS A NEW LINE must still be
        # honored. The breaker exists to kill questions that keep circling the dead
        # topic — not to silence someone who just pivoted to a new subject. A
        # rotating-loop question is semantically close to the saturated window
        # (low novelty even when the wording/names differ), whereas a genuine new
        # question is novel against it. So we only redirect when the last utterance
        # is NOT a fresh, on-a-new-topic direct question.
        if repetition_tracker is not None and available:
            saturated, _conc, _nov = repetition_tracker.recent_topic_saturation()
            if saturated and not self._last_is_fresh_direct_question(
                last_utterance, available, repetition_tracker
            ):
                redirect = self._redirect_for_saturation(
                    state, thoughts, available, last_speaker, repetition_tracker
                )
                if redirect is not None:
                    return redirect

        # Detect whether the last utterance is a nagging REPEAT — i.e. the speaker
        # already asked a semantically equivalent question earlier (it is now "in the
        # history"). A repeated question must not force the addressee to answer the
        # same thing again, and the asker must not be allowed to keep the floor to
        # re-ask it. This is the penalty for redundant questioning.
        last_is_repeat_question = False
        last_repeat_match = None
        if (
            last_utterance
            and repetition_tracker is not None
            and last_utterance.get("is_question")
        ):
            # Use the NEAR-VERBATIM threshold, not the looser novelty threshold:
            # interrogation questions share so much structure ("did you take X
            # before 9:00?", "were you in the apartment?") that the 0.82 novelty
            # cutoff flags almost every question as a repeat. Only a near-duplicate
            # of the asker's own earlier question should count as nagging.
            is_repeat, _sim, last_repeat_match = repetition_tracker.last_utterance_repeats_earlier(
                same_speaker_only=True,
                threshold=repetition_tracker.high_sim_threshold,
            )
            last_is_repeat_question = is_repeat

        # FIRST: Check for direct address using explicit pattern matching, then
        # fall back to an LLM judgement for ambiguous question + name combinations
        # (e.g., "Margaret, did you notice anyone slip into the bathroom?").
        if last_utterance:
            directly_addressed = detect_direct_address(last_utterance["text"], available)
            if not directly_addressed:
                directly_addressed = detect_direct_address_llm(
                    self.llm,
                    last_utterance["text"],
                    available,
                    last_speaker=last_speaker,
                )
            if directly_addressed:
                # A repeated question only suppresses the mandatory response in the
                # genuine "nagging" case: the asker is re-asking something the
                # addressee has ALREADY answered. If the addressee never answered the
                # earlier equivalent question (e.g. a fresh, pointed question that
                # merely resembles wording the asker used before), they must still
                # respond — stripping the mandate here would silence a legitimate
                # direct question and let the floor drift to a redundant re-asker.
                suppress_for_nag = False
                if last_is_repeat_question and last_repeat_match is not None:
                    # Only genuine nagging: the addressee already gave a MANDATED
                    # answer to this same asker after the earlier equivalent question
                    # was raised. "Spoke at some point later" is far too broad in late
                    # rounds (everyone has spoken many times) and would silence a
                    # legitimate fresh question that merely shares interrogation phrasing.
                    suppress_for_nag = any(
                        u.get("speaker") == directly_addressed
                        and u.get("response_to_speaker") == last_speaker
                        and u.get("turn", -1) > last_repeat_match.turn
                        for u in history
                    )
                if not suppress_for_nag:
                    question_text = last_utterance.get("text", "")[:300]
                    return SpeakerDecision(
                        reasoning=f"{directly_addressed} was directly addressed by {last_speaker}",
                        next_speaker=directly_addressed,
                        response_constraint=f"{last_speaker} asked you: \"{question_text}\" — answer this specific question first, then you may add anything else.",
                        is_direct_address=True
                    )
        
        # SECOND: Self-selection by bid strength among available agents
        available_thoughts = {name: tr for name, tr in thoughts.items() if name in available}

        # Identify agents who have contributed the fewest novel facts this round
        # (used for a low-contribution boost below)
        current_round = state.get("current_round", 1)
        low_contribution_agents: list = []
        if repetition_tracker is not None:
            low_contribution_agents = repetition_tracker.get_low_contribution_agents(
                all_agents=self.agent_names, round_num=current_round, top_n=max(1, len(self.agent_names) // 2)
            )

        speaking_bids = {}
        adjusted_scores: dict = {}
        for name, tr in available_thoughts.items():
            if not (tr.action == "speak" and tr.importance >= 7):
                continue
            thought_text = getattr(tr, "thought", "")

            # Novelty adjustment: semantic similarity against recent utterances
            # (replaces the old token-overlap heuristic)
            if repetition_tracker is not None and thought_text.strip():
                novelty = max(0.35, repetition_tracker.novelty_score(thought_text, against_last_n=8))
            else:
                # Fallback: token-overlap against last 6 utterances
                recent_msgs = " ".join([u.get("text", "") for u in history[-6:]]).lower()
                thought_tokens = set(t for t in thought_text.lower().split() if len(t) > 2)
                recent_tokens = set(t for t in recent_msgs.split() if len(t) > 2)
                overlap = float(len(thought_tokens & recent_tokens)) / (len(thought_tokens) or 1)
                novelty = max(0.35, 1.0 - overlap)

            adjusted_importance = int(round(tr.importance * novelty))

            # Low-contribution boost: agents with few novel turns get +1 so they
            # are preferred in ties, nudging them to share more of their knowledge
            if name in low_contribution_agents:
                adjusted_importance = min(9, adjusted_importance + 1)

            speaking_bids[name] = tr
            adjusted_scores[name] = adjusted_importance

        if speaking_bids:
            # Use adjusted_importance when available, fall back to original importance
            sorted_by_bid = sorted(
                speaking_bids.items(),
                key=lambda x: adjusted_scores.get(x[0], x[1].importance),
                reverse=True,
            )
            highest_score = adjusted_scores.get(sorted_by_bid[0][0], sorted_by_bid[0][1].importance)
            top_agents = [name for name, tr in sorted_by_bid if adjusted_scores.get(name, tr.importance) == highest_score]

            if len(top_agents) == 1:
                winner = top_agents[0]
                reason_type = getattr(speaking_bids[winner], "reason_type", "bid")
                reasoning = f"{winner} has the strongest self-selection bid ({highest_score}/9, reason={reason_type})"
                return SpeakerDecision(
                    reasoning=reasoning,
                    next_speaker=winner,
                    response_constraint=None,
                    is_direct_address=False
                )

            available_str = ", ".join(top_agents)
            agent_thoughts_txt = "\n".join([
                f"- {name}: bid={speaking_bids[name].importance}/9, reason={getattr(speaking_bids[name], 'reason_type', 'bid')}, thinking: \"{speaking_bids[name].thought}\""
                for name in top_agents
            ])
        else:
            # THIRD: nobody bid strongly. The last speaker is NEVER allowed to keep
            # the floor here — `available` already excludes them, and handing the turn
            # back produces exactly the monopolization / verbatim-repeat loop we want
            # to prevent (same person speaking twice in a row). Always force the turn
            # to a different agent and let the GM tie-break choose who takes the floor.
            available_str = ", ".join(available)
            agent_thoughts_txt = "\n".join([
                f"- {name}: bid={available_thoughts[name].importance}/9, reason={getattr(available_thoughts[name], 'reason_type', 'no_move')}, thinking: \"{available_thoughts[name].thought}\""
                for name in available
            ]) if available_thoughts else "(no strong self-selection bids available)"
            top_agents = available
        
        # Build conversation history for context
        history_txt = "\n".join([
            f"{u['speaker']}: {u['text']}" for u in history[-5:]  # Last 5 messages for context
        ]) or "(no conversation yet)"
        
        coverage_note = ""
        if low_contribution_agents:
            coverage_note = (
                f"\nKNOWLEDGE COVERAGE: These players have contributed the fewest novel facts so far "
                f"and should be preferred if their bid is otherwise comparable: "
                f"{', '.join(low_contribution_agents)}."
            )

        msgs = [
            SystemMessage(content=f"""{self.persona}

These players have EQUAL urgency scores and are tied: {available_str}
You must break the tie by choosing who would best advance the murder investigation.
Prefer players who have not yet shared their private knowledge or who have been silent.
Return valid JSON with keys: reasoning, next_speaker, response_constraint, is_direct_address."""),
            HumanMessage(content=f"""RECENT CONVERSATION:
{history_txt}

TIED PLAYERS' THOUGHTS:
{agent_thoughts_txt}
{coverage_note}
Choose ONE player from the tied players to speak next: {available_str}
Return JSON only."""),
        ]
        
        try:
            result = self.llm_decide.invoke(msgs)
            
            # Validate the chosen speaker is in the tied group
            if result.next_speaker not in top_agents:
                # Try to find a close match
                for agent in top_agents:
                    if agent.lower() in result.next_speaker.lower() or result.next_speaker.lower() in agent.lower():
                        result.next_speaker = agent
                        break
                else:
                    # Default to first tied agent
                    result.next_speaker = top_agents[0]
            
            result.reasoning = f"Tie-breaker: {result.reasoning}"
            # The tie-break is a self-selection among bidders, never a mandated reply.
            # Don't let the LLM mislabel it as a direct address (which would print the
            # "must respond" tag and create a spurious pending obligation).
            result.is_direct_address = False
            result.response_constraint = None
            return result
        except Exception as e:
            print(f"Error in GameMaster decide: {e}")
            # Fallback: pick highest urgency
            max_urgency = max(thoughts.items(), key=lambda x: x[1].importance)
            return SpeakerDecision(
                reasoning="Fallback selection based on urgency",
                next_speaker=max_urgency[0],
                response_constraint=None,
                is_direct_address=False
            )

    def _last_is_fresh_direct_question(
        self, last_utterance, available: list, repetition_tracker
    ) -> bool:
        """True if the last utterance is a direct question that opens a NEW line.

        Used to shield a legitimate pointed question from the topic-saturation
        redirect. "Fresh" means it directly addresses an available agent AND is
        semantically novel relative to the recent (saturated) window — i.e. it
        pivots the conversation rather than re-circling the dead topic. A
        rotating-interrogation question stays close to the window and is therefore
        NOT fresh, so the breaker still fires on it.
        """
        if not last_utterance or not last_utterance.get("is_question"):
            return False
        text = last_utterance.get("text", "")
        if not text.strip():
            return False
        addressee = detect_direct_address(text, available)
        if not addressee:
            return False
        # Compare this question against the turns BEFORE it (the saturated window).
        # The tracker calls an utterance novel when its similarity to prior turns is
        # below novelty_threshold, i.e. novelty above (1 - novelty_threshold); reuse
        # that same cutoff so "opens a new line" matches the tracker's own notion of
        # novelty. last_utterance_novelty excludes the question itself from the
        # comparison (it is already stored), avoiding a spurious self-match.
        novelty = repetition_tracker.last_utterance_novelty(window=6)
        return novelty >= (1.0 - repetition_tracker.novelty_threshold)

    def _redirect_for_saturation(
        self, state: dict, thoughts: dict, available: list, last_speaker, repetition_tracker
    ) -> Optional[SpeakerDecision]:
        """Pick a quiet agent and steer the group off a saturated topic.

        Selection: among the lowest-contribution available agents, prefer whoever's
        current thought is already the furthest from the recent (saturated) window —
        i.e. the person most ready to talk about something else. Falls back to the
        quietest available agent. Returns None only if there is genuinely no one to
        redirect to, in which case normal selection proceeds.
        """
        current_round = state.get("current_round", 1)
        ranked = repetition_tracker.get_low_contribution_agents(
            all_agents=self.agent_names, round_num=current_round, top_n=len(self.agent_names)
        )
        candidates = [n for n in ranked if n in available] or list(available)
        if not candidates:
            return None

        # Among the quietest candidates, prefer the one whose pending thought is
        # most novel relative to the recent window (most likely to change subject).
        best, best_novelty = None, -1.0
        for name in candidates:
            tr = thoughts.get(name)
            thought_text = getattr(tr, "thought", "") if tr is not None else ""
            novelty = (
                repetition_tracker.novelty_score(thought_text, against_last_n=6)
                if thought_text.strip()
                else 0.0
            )
            if novelty > best_novelty:
                best, best_novelty = name, novelty
        shifter = best or candidates[0]

        stopwords = getattr(self.scenario, "gate_stopwords", None)
        dead_topic = repetition_tracker.dominant_recent_term(window=6, stopwords=stopwords)
        topic_clause = f'"{dead_topic}"' if dead_topic else "the current line of questioning"

        constraint = (
            f"The discussion has stalled — the last several turns circled {topic_clause} "
            f"without revealing anything new. Do NOT ask about it again. Open a NEW line of "
            f"inquiry: probe motive, the timeline, a contradiction in someone's earlier alibi, "
            f"or a clue the group hasn't examined yet."
        )
        return SpeakerDecision(
            reasoning=(
                f"Topic saturation detected ({topic_clause}); demoting the direct-address "
                f"mandate and giving {shifter} the floor to open a new line of inquiry"
            ),
            next_speaker=shifter,
            response_constraint=constraint,
            is_direct_address=False,
        )

    def provide_initial_context(self) -> str:
        """Provide game context to all players at the start."""
        investigation_rounds = f"Rounds 2-{self.max_rounds - 1}" if self.max_rounds > 2 else "Round 2"
        accusation_round = self.max_rounds
        title = self.scenario.title.upper()
        context_intro = f"""
{_boxed(title, style="double")}

TRAGEDY HAS STRUCK!

{self.scenario.victim_status_line}

{self.scenario.introduction_text}

══════════════════════════════════════════════════════════════════

SUSPECTS PRESENT: {', '.join(self.agent_names)}

══════════════════════════════════════════════════════════════════

GAME STRUCTURE:
- Round 1: Introductions - Each suspect introduces themselves
- {investigation_rounds}: Investigation - Question each other, find the killer!
- Round {accusation_round}: Accusation - Each person accuses someone
- Final: Confessions - Everyone reveals their secrets

Each round has approximately {self.conversations_per_round} conversations.

══════════════════════════════════════════════════════════════════

YOUR GOAL: {self.scenario.investigation_goal}

══════════════════════════════════════════════════════════════════

RULES:
1. Everyone must participate - silence makes you suspicious!
2. Ask questions, share clues, and make accusations
3. All conversations are PUBLIC - no private discussions
4. Players CANNOT accuse themselves
5. After round {self.max_rounds - 1}, everyone votes on who they think is the murderer

══════════════════════════════════════════════════════════════════

NOW BEGINNING ROUND 1: INTRODUCTIONS...
Let each suspect introduce themselves to the group.

"""
        return context_intro
    
    def announce_round_change(self, new_round: int) -> str:
        """Generate announcement for round change, including clues."""
        # Clue number trails the round by two (Clue #1 first appears in Round 3);
        # see get_clue_for_round for the full schedule.
        clue_number = new_round - 2
        clue_text = self._load_clue(clue_number) if clue_number >= 1 else ""
        
        clue_section = ""
        if clue_text:
            clue_section = f"""
{_boxed("NEW CLUE DISCOVERED!", style="single")}

{clue_text}

══════════════════════════════════════════════════════════════════
"""

        if new_round == 2:
            return f"""
{_boxed(f"ROUND {new_round}: THE INVESTIGATION BEGINS", style="double")}

The introductions are complete. Now the real investigation begins!
{clue_section}
Remember: {self.scenario.victim_name} was murdered.
One of you is the killer. Question everyone. Look for lies and
inconsistencies. {self.scenario.investigation_goal}
"""
        elif new_round < self.max_rounds:
            return f"""
{_boxed(f"ROUND {new_round}", style="double")}

New evidence has emerged!
{clue_section}
The truth about {self.scenario.victim_name}'s murder is getting closer...
Continue questioning. The killer is among you!
"""
        else:
            # The decisive clues (#4, #5) were revealed and debated in the final
            # DISCUSSION round (max_rounds - 1); this banner is shown by
            # _complete_investigation immediately before the accusation step. Do NOT
            # embed clue text here — there is no more discussion, and re-revealing
            # would duplicate evidence already on the table.
            return f"""
{_boxed("FINAL ROUND - TIME TO ACCUSE", style="double")}

The investigation is complete.

It's time to decide: {self.scenario.accusation_prompt}
Each of you must now make your final accusation!
"""
