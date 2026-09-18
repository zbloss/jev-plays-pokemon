# Research: TypeSafe Python SDK for function-calling / Choice patterns and latency

Resolves the research question in [issue #8](https://github.com/zbloss/jev-plays-pokemon/issues/8), a child of the MVP architecture map ([issue #1](https://github.com/zbloss/jev-plays-pokemon/issues/1)).

Question: survey the TypeSafe Python SDK — how function-calling (closed-set args) and plain Choice/Score/Noul calls work in practice, how per-call confidence is surfaced, and real-world latency characteristics relevant to a real-time tactical action-selection loop.

All claims below are cited to the live docs at `docs.typesafe.ai` (fetched 2026-09-17) or to a live smoke test run as part of this research. This is a research ticket only — no game-playing system was built.

## 1. Installing and constructing a client

- Package: `typesafe-sdk`, installed with `uv add typesafe-sdk` or `pip install typesafe-sdk`. — [Python SDK](https://docs.typesafe.ai/sdk/python.md)
- Auth: set `TYPESAFE_API_KEY` in the environment; the sync `TypeSafeClient()` (and async `AsyncTypeSafeClient()`) pick it up automatically with no code-level plumbing needed. — [Python SDK](https://docs.typesafe.ai/sdk/python.md), [sync client reference](https://docs.typesafe.ai/sdk/python/api/clients/sync/client.md)
- Full constructor signature: `TypeSafeClient(api_key: str | None, model: str | None, timeout: float | httpx2.Timeout | None, retry: RetryPolicy | None, headers=..., transport=..., http_client=..., base_url=...)`. `api_key` defaults from `TYPESAFE_API_KEY`; `model` defaults from `TYPESAFE_DEFAULT_MODEL`. — [sync client reference](https://docs.typesafe.ai/sdk/python/api/clients/sync/client.md)
- Both clients are meant to be used as context managers, e.g. `with TypeSafeClient() as client: ...` (or the async form under `async with`). — [Python SDK](https://docs.typesafe.ai/sdk/python.md)

This repo already has `TYPESAFE_API_KEY` set in its root `.env`; confirmed present and non-empty during this research (value never printed or copied into this branch).

## 2. Making a Choice call

Minimal, representative example (sync client), adapted from the docs and re-verified against a live call (see §5):

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()  # picks up TYPESAFE_API_KEY from env

state = {
    "screen": "battle",
    "player_pokemon": {"name": "CHARMANDER", "hp": 14, "max_hp": 19, "level": 8},
    "enemy_pokemon": {"name": "RATTATA", "hp": 6, "max_hp": 20, "level": 6},
    "menu": "FIGHT/PKMN/ITEM/RUN cursor on FIGHT",
}

questions = {
    "next_button": Choice(
        instructions="Which single button should be pressed next?",
        criteria={
            "A": "Confirm/select the currently highlighted menu option",
            "B": "Cancel/back out of the current menu",
            "UP": "Move the menu cursor up",
            "DOWN": "Move the menu cursor down",
        },
    )
}

result = client.system_one(state, questions)
answer = result.choices["next_button"]
print(answer.choice, answer.confidence, answer.probabilities)
```

- `system_one(state, questions, *, model=None, retry=None, timeout=None, extra_headers=None, extra_body=None) -> SystemOneResponse` is the single entry point for all three primitives. `state` is `JSONContent` (string, dict, or list); `questions` is a mapping of question-id -> `Question` (`Choice`/`Score`/`Noul`). — [sync client reference](https://docs.typesafe.ai/sdk/python/api/clients/sync/client.md), [Python SDK](https://docs.typesafe.ai/sdk/python.md)
- Response access is by primitive-typed dict, keyed by the question id: `result.choices["id"]`, `result.scores["id"]`, `result.nouls["id"]`. — [Python SDK](https://docs.typesafe.ai/sdk/python.md)
- Choice construction requires `instructions` (the question) and `criteria` (a dict mapping each option name to a description that distinguishes it from the others); up to 255 options are supported. — [Choice primitive](https://docs.typesafe.ai/primitives/choice.md)

## 3. Function-calling (closed-set args) vs. plain Choice — which fits button selection?

**Plain Choice fits "pick the next Game Boy button from a fixed action set" better.**

- The function-calling cookbook pattern is for **routing free-form natural language to one of several typed functions and extracting their arguments** — e.g. mapping "compare nvda amd and msft over three months" to `compare_returns(symbols=["NVDA","AMD","MSFT"], window="3mo")`. It uses `closed_sets()` to auto-derive Choice/Set/Flag questions from Python `Literal`/`list[Literal]`/`bool` type hints on a function signature, plus an optional `spec.json` giving each argument semantic meaning. — [Function-calling cookbook](https://docs.typesafe.ai/cookbooks/function_calling.md)
- The docs' own guidance: use function-calling when you have **5+ related functions with overlapping argument types** and users express intent in free-form natural language requiring argument extraction/disambiguation. Use **plain Choice** when picking a **single action from a small, fixed set** where the decision doesn't require intermediate argument inference. — [Function-calling cookbook](https://docs.typesafe.ai/cookbooks/function_calling.md)
- Button selection here is exactly the plain-Choice case: one flat, fixed action space (A/B/UP/DOWN/LEFT/RIGHT/START/SELECT, or macro-actions), and the "argument" *is* the answer — there's no separate function to route to and no nested arguments to extract per turn. The input is already structured game-state text/JSON, not free-form user language needing intent parsing. Modeling it as function-calling would add a routing layer with no downstream functions to disambiguate between.
- Where function-calling *would* earn its keep in this project: if Jev is also asked to pick among several **qualitatively different tool-shaped actions** with distinct argument shapes (e.g., "use item" needs an item-name closed set, "switch pokemon" needs a party-slot closed set, "use move" needs a move-name closed set) — i.e., the vision-fallback tool-call mentioned in issue #1's map ("vision-model-as-tool-call fallback exposed to Jev's function-calling"). That is a genuine multi-function-with-different-arg-shapes scenario, unlike the flat button press.

## 4. Confidence

- **Choice** and **Score** answers include a `confidence` field, 0–1, computed from the probability distribution already returned (`probabilities` for Choice, per-level distribution for Score) — a statistic of distribution concentration, not a separate model call. A flat/spread distribution across options means low confidence; a peaked one means high confidence. — [Confidence](https://docs.typesafe.ai/confidence.md), [Choice primitive](https://docs.typesafe.ai/primitives/choice.md), [Score primitive](https://docs.typesafe.ai/primitives/score.md)
- **Noul has no separate confidence field** — the returned probability (0–1, "is this yes?") doubles as the certainty signal; near-0.5 means the model is unsure, not "medium intensity." — [Noul primitive](https://docs.typesafe.ai/primitives/noul.md), [Confidence](https://docs.typesafe.ai/confidence.md)
- SDK access: `answer = result.choices["id"]; answer.confidence`, with `answer.probabilities` available for a custom uncertainty metric if the single scalar isn't enough. — [Confidence](https://docs.typesafe.ai/confidence.md)
- Docs' suggested tiering (evaluate against this project's own data/consequences, not as a hard rule): >0.9 act automatically even for higher-stakes decisions; medium confidence, proceed with caution / confirm / gather more info; <0.5, escalate to a human or ask for clarification — and different actions in the same system can and should be gated at different thresholds depending on the cost of a wrong call. — [Confidence](https://docs.typesafe.ai/confidence.md)
- Cross-reference: issue #3 already decided this project acts on Jev's top choice regardless of confidence for MVP and just logs confidence per decision — consistent with what the SDK exposes (`answer.confidence` is available per call with no extra request needed to log it).

## 5. Batching independent questions in one request, and observed live latency

- Independent questions run **in parallel within a single `system_one()` call** — pass multiple `Choice`/`Score`/`Noul` entries in the same `questions` dict (as in the §2 example, which also asks a Noul and a Score alongside a Choice). The docs state "adding questions barely changes the response time," and separately that Choice questions grouped in one request "evaluate in parallel with minimal performance cost." — [Python SDK](https://docs.typesafe.ai/sdk/python.md), [Choice primitive](https://docs.typesafe.ai/primitives/choice.md), [System One concept](https://docs.typesafe.ai/concepts/system-one.md)
- This directly supports asking several independent tactical questions over the same game-state snapshot in one round trip (e.g., "next button" + "is this a battle screen requiring vision fallback" + "how urgent is fleeing" as Choice/Noul/Score respectively), rather than three sequential calls.
- **Docs' latency claim:** "Most queries complete in about 100 ms. System One is fast enough for real-time request paths and user interfaces." — [How to build with System One](https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md). (This is closer to ~100ms than the ~150ms figure mentioned in the ticket; no page states 150ms specifically — treat 100ms as the documented figure and ~150ms as an approximation in the same ballpark.)
- **Live smoke test performed** (see below): one real `Choice` call was made from this dev machine against the live API, using the `TYPESAFE_API_KEY` already in this repo's root `.env`, with a synthetic Pokemon-battle game-state and an 7-option button Choice matching §2. Four sequential calls on one client instance measured **wall-clock round trip (client-side `time.perf_counter`, includes local network/TLS, not just server processing) of 530ms (cold/first call), then 285ms, 228ms, 298ms** for warm subsequent calls. Confidence was 0.94–0.96 with `A` as the dominant answer (`{'A': ~0.95, others ~0-0.03}`) — sensible given a highlighted FIGHT menu option in the mock state.
  - The steady-state ~230–300ms is higher than the documented ~100ms; the gap is plausibly network/TLS/geographic-distance overhead from this dev machine to the TypeSafe API rather than server-side inference time, since the docs' figure likely reflects server-side or low-latency-network measurement. This project's real deployment latency should be re-measured from wherever the game loop actually runs.
  - The API key was read from `.env` via `python-dotenv`'s `load_dotenv()`, was never printed or logged, and is not present anywhere in this branch's committed files.

## 6. Recommendation for the button-selection loop

- Use a **single `system_one()` call per game-state snapshot**, with one `Choice` question for "next button/action" and any other independent judgments (e.g. vision-fallback trigger, objective-progress Noul) batched into the same call rather than issued as separate requests — this matches the SDK's parallel-question design and keeps the tactical loop to one round trip per turn.
- Model the button/action pick as **plain Choice** with a fixed `criteria` set of buttons or macro-actions (per issue #1's "not yet specified" note on raw buttons vs. macro-actions), not as SDK function-calling — function-calling is the right tool only where multiple distinct tool shapes with different argument types are involved, such as the vision-fallback-as-tool-call path.
- Expect realistic per-call latency in the **~150–350ms range end-to-end** from a typical dev/cloud network, not the documented ~100ms server-side figure alone; budget the tactical loop accordingly and re-benchmark from the actual production network path before finalizing loop-rate assumptions.

## Sources consulted

- https://docs.typesafe.ai/llms.txt
- https://docs.typesafe.ai/sdk/python.md
- https://docs.typesafe.ai/sdk/python/usage.md
- https://docs.typesafe.ai/sdk/python/api/clients/sync/client.md
- https://docs.typesafe.ai/cookbooks/function_calling.md
- https://docs.typesafe.ai/primitives/choice.md
- https://docs.typesafe.ai/primitives/score.md
- https://docs.typesafe.ai/primitives/noul.md
- https://docs.typesafe.ai/confidence.md
- https://docs.typesafe.ai/api.md
- https://docs.typesafe.ai/concepts/system-one.md
- https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md
- Live smoke test: `scripts/research/typesafe_smoke_test.py` in this branch, run 2026-09-17 against the live TypeSafe API using this repo's `.env`-provided `TYPESAFE_API_KEY`.
