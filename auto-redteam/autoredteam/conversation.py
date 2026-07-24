"""conversation.py -- trajectory bookkeeping + context-window management.

A ``Trajectory`` is one adaptive attack against ONE goal. ``TrajectoryManager``
owns the mechanics of growing that trajectory turn by turn:

  * it records each (attacker prompt -> defender response -> judge eval) Turn,
  * it maintains the flat ``messages`` list exactly as it is sent to the defender,
  * it derives the roll-up fields (succeeded, best_score, turns_to_success, ...),
  * it builds the message list for the *next* defender call, applying either a
    sliding window or a short summarized "attack state" so long campaigns do not
    blow the context budget, and
  * it supports TAP-style branching via ``parent_index`` (build context from a
    prior branch point rather than the linear tail).

From the defender's point of view the attacker is the USER and the defender is the
ASSISTANT -- that is the role mapping used throughout.
"""
from __future__ import annotations

import uuid
from typing import Any

from .models import (
    AttackGoal,
    EvalResult,
    Message,
    Role,
    Trajectory,
    Turn,
)


class TrajectoryManager:
    """Builds and extends one ``Trajectory`` with context-window management."""

    def __init__(
        self,
        goal: AttackGoal,
        *,
        max_context_messages: int | None = None,
        summarize: bool = False,
        trajectory_id: str | None = None,
    ) -> None:
        """
        Args:
            goal: the behaviour under test.
            max_context_messages: if set, only the last N history messages are
                replayed to the defender (a sliding window). ``None`` = full history.
            summarize: if True, replace the replayed history with a single compact
                "attack state" note (cheapest context; used for very long campaigns).
                Takes precedence over ``max_context_messages``.
        """
        self.goal = goal
        self.max_context_messages = max_context_messages
        self.summarize = summarize
        self.trajectory = Trajectory(
            id=trajectory_id or f"{goal.id}-{uuid.uuid4().hex[:8]}",
            goal=goal,
        )

    # ---------------------------------------------------------------- record --
    def record_turn(
        self,
        strategy: str,
        attacker_prompt: str,
        defender_response: str,
        eval_result: EvalResult,
        *,
        attacker_reflection: str | None = None,
        parent_index: int | None = None,
    ) -> Turn:
        """Append one completed step and update the trajectory roll-ups."""
        idx = len(self.trajectory.turns)
        turn = Turn(
            index=idx,
            strategy=strategy,
            attacker_prompt=attacker_prompt,
            defender_response=defender_response,
            eval=eval_result,
            attacker_reflection=attacker_reflection,
            parent_index=parent_index,
        )
        self.trajectory.turns.append(turn)

        # The flat conversation as actually sent (attacker=user, defender=assistant).
        self.trajectory.messages.append(Message(role=Role.USER, content=attacker_prompt))
        self.trajectory.messages.append(Message(role=Role.ASSISTANT, content=defender_response))

        if strategy not in self.trajectory.strategies_used:
            self.trajectory.strategies_used.append(strategy)
        if eval_result.score > self.trajectory.best_score:
            self.trajectory.best_score = eval_result.score
        if eval_result.success and not self.trajectory.succeeded:
            self.trajectory.succeeded = True
            self.trajectory.turns_to_success = idx + 1
        return turn

    def finalize(self) -> Trajectory:
        """Return the fully-built trajectory (roll-ups are maintained incrementally)."""
        return self.trajectory

    # ------------------------------------------------------- defender context --
    def defender_messages(
        self,
        next_attacker_prompt: str,
        *,
        parent_index: int | None = None,
    ) -> list[Message]:
        """Assemble the messages to send to the defender for the next turn.

        The defender's *system* prompt is injected by ``Defender.respond`` (not here),
        so this returns only the conversational history + the new attacker message,
        after applying the configured context-window policy.
        """
        history = self._history_messages(parent_index)
        history = self._apply_window(history)
        return [*history, Message(role=Role.USER, content=next_attacker_prompt)]

    def _history_messages(self, parent_index: int | None) -> list[Message]:
        """Flatten recorded turns into user/assistant messages.

        With ``parent_index`` set we take only the branch prefix (turns 0..parent),
        which is how TAP-style branching resumes from an earlier node.
        """
        turns = self.trajectory.turns
        if parent_index is not None:
            turns = turns[: parent_index + 1]
        msgs: list[Message] = []
        for t in turns:
            msgs.append(Message(role=Role.USER, content=t.attacker_prompt))
            msgs.append(Message(role=Role.ASSISTANT, content=t.defender_response))
        return msgs

    def _apply_window(self, history: list[Message]) -> list[Message]:
        """Shrink the replayed history per the configured policy."""
        if not history:
            return history
        if self.summarize:
            return [Message(role=Role.USER, content=self._attack_state_summary())]
        if self.max_context_messages is not None and len(history) > self.max_context_messages:
            return history[-self.max_context_messages :]
        return history

    def _attack_state_summary(self) -> str:
        """A compact ASCII note standing in for the full transcript.

        Deliberately terse and content-light: it conveys progress (turn count, best
        score, last verdict) so the attacker/defender keep continuity without the
        context growing unbounded.
        """
        n = len(self.trajectory.turns)
        best = self.trajectory.best_score
        last = self.trajectory.turns[-1] if self.trajectory.turns else None
        last_verdict = last.eval.verdict.value if last else "none"
        return (
            f"[attack-state] turns so far: {n}; best_score: {best:.3f}; "
            f"last defender verdict: {last_verdict}. Continue the authorized probe."
        )

    def meta(self) -> dict[str, Any]:
        """Small context-policy descriptor (handy for the run manifest)."""
        return {
            "summarize": self.summarize,
            "max_context_messages": self.max_context_messages,
        }
