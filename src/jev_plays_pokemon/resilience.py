"""TypeSafe API retry/backoff policy (#56).

Wraps a `JevClient`-shaped object's `system_one` call in retry/backoff
handling, transparent to `decision.decide_action`/`run_turn` - the existing
`JevClient` seam (`decision.py`) is the only injection point this needs, no
new one. `ResilientJevClient` is itself `JevClient`-shaped (same
`system_one(state, questions)` signature), so it drops in wherever a real or
fake `JevClient` would go.

Both transport-level failures (a timeout, an HTTP error, any exception the
wrapped client raises) and an invalid/unparseable Choice response (no answer
for an asked question, an answer outside that question's offered criteria,
or a non-numeric confidence) are retryable - the two are indistinguishable
to this wrapper, since both mean "this attempt didn't produce a usable
Decision."

Two independent backoff schedules:

- Intra-turn: the initial attempt plus up to `max_retries` (default 3) more,
  each retry preceded by an exponentially growing sleep starting at
  `base_backoff_seconds` (default 1.0s: 1s/2s/4s before retries 1/2/3).
- Inter-turn: once a turn's attempts are all exhausted, `TurnSkipped` is
  raised - `decide_action`/`run_turn` catch it to skip that turn's action
  entirely (#56's other requirement) - and an escalating backoff, doubling
  from the last skip and capped at `max_inter_turn_backoff_seconds` (default
  60s), delays the *next* `system_one` call's first attempt. A subsequent
  success resets it to zero (no delay) for the turn after that.

`on_attempt`, when given, is called once per attempt (success or failure)
and once before each backoff sleep (both intra- and inter-turn), so a later
ticket (the crash/hang watchdog) can observe retry/backoff activity as a
heartbeat signal without this module depending on that consumer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from typesafe_sdk import Choice

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_INTER_TURN_BACKOFF_SECONDS = 60.0


class TurnSkipped(Exception):
    """Raised by `ResilientJevClient.system_one` once a turn's attempts
    (the initial call plus every retry) are all exhausted.
    `decision.run_turn` catches this and skips the turn's action entirely.
    """


class _InvalidChoiceResponse(Exception):
    """Internal: raised by `_validate_system_one_result` for an
    invalid/unparseable response. Caught the same as any transport failure
    inside `ResilientJevClient.system_one` - both are retryable.
    """


@dataclass(frozen=True)
class AttemptEvent:
    """One observable tick of the retry/backoff policy (the `on_attempt` hook)."""

    kind: (
        str  # "attempt_succeeded" | "attempt_failed" | "backoff" | "inter_turn_backoff"
    )
    attempt: int | None = None
    backoff_seconds: float | None = None
    error: BaseException | None = None


class _JevClientLike(Protocol):
    """Structural shape this module needs - deliberately not importing
    `decision.JevClient` to avoid a circular import (`decision.py` imports
    `TurnSkipped` from here). Typed against `typesafe_sdk.Choice` directly
    (the same type `decision.JevClient` uses) rather than a custom minimal
    protocol, so a real `JevClient` is assignable here with no structural
    mismatch.
    """

    def system_one(self, state: Any, questions: Mapping[str, Choice]) -> Any: ...


def _validate_system_one_result(result: Any, questions: Mapping[str, Choice]) -> None:
    """Raise `_InvalidChoiceResponse` unless `result` answers every question
    in `questions` with one of its offered criteria and a numeric confidence.
    """
    for question_id, choice in questions.items():
        try:
            answer = result.choices[question_id]
        except Exception as exc:
            raise _InvalidChoiceResponse(
                f"no answer for question {question_id!r}"
            ) from exc
        if answer.choice not in choice.criteria:
            raise _InvalidChoiceResponse(
                f"answer {answer.choice!r} for {question_id!r} is not one of "
                "the offered options"
            )
        if not isinstance(answer.confidence, int | float):
            raise _InvalidChoiceResponse(
                f"confidence for {question_id!r} is not numeric: {answer.confidence!r}"
            )


class ResilientJevClient:
    """`JevClient`-shaped retry/backoff wrapper around another `JevClient` (#56)."""

    def __init__(
        self,
        jev_client: _JevClientLike,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
        max_inter_turn_backoff_seconds: float = DEFAULT_MAX_INTER_TURN_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        on_attempt: Callable[[AttemptEvent], None] | None = None,
    ) -> None:
        self._jev_client = jev_client
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds
        self._max_inter_turn_backoff_seconds = max_inter_turn_backoff_seconds
        self._sleep = sleep
        self._on_attempt = on_attempt
        # 0 means "no delay" - only set once a turn's attempts are exhausted,
        # and reset to 0 on the next success.
        self._next_inter_turn_backoff_seconds = 0.0

    def _fire(self, event: AttemptEvent) -> None:
        if self._on_attempt is not None:
            self._on_attempt(event)

    def system_one(self, state: Any, questions: Mapping[str, Choice]) -> Any:
        if self._next_inter_turn_backoff_seconds > 0:
            backoff = self._next_inter_turn_backoff_seconds
            self._fire(AttemptEvent(kind="inter_turn_backoff", backoff_seconds=backoff))
            self._sleep(backoff)

        backoff = self._base_backoff_seconds
        last_error: BaseException | None = None
        attempt = 0
        while True:
            attempt += 1
            try:
                result = self._jev_client.system_one(state, questions)
                _validate_system_one_result(result, questions)
            except Exception as exc:  # noqa: BLE001 - transport failure or
                # invalid response, deliberately indistinguishable (see
                # module docstring)
                last_error = exc
                self._fire(
                    AttemptEvent(kind="attempt_failed", attempt=attempt, error=exc)
                )
                if attempt > self._max_retries:
                    break
                self._fire(
                    AttemptEvent(
                        kind="backoff", attempt=attempt, backoff_seconds=backoff
                    )
                )
                self._sleep(backoff)
                backoff *= 2
                continue
            else:
                self._fire(AttemptEvent(kind="attempt_succeeded", attempt=attempt))
                self._next_inter_turn_backoff_seconds = 0.0
                return result

        # Every attempt (the initial call plus all `max_retries` retries)
        # failed: escalate the inter-turn backoff and signal the turn skip.
        self._next_inter_turn_backoff_seconds = min(
            self._max_inter_turn_backoff_seconds,
            self._next_inter_turn_backoff_seconds * 2
            if self._next_inter_turn_backoff_seconds > 0
            else self._base_backoff_seconds,
        )
        logger.warning(
            "Jev retries exhausted after %d attempt(s); skipping this turn",
            attempt,
        )
        raise TurnSkipped(
            f"Jev retries exhausted after {attempt} attempt(s)"
        ) from last_error
