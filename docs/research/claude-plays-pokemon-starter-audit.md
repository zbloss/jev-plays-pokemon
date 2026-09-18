# Audit: ClaudePlaysPokemonStarter (reuse vs. must-replace for Jev)

Research for [issue #6](https://github.com/zbloss/jev-plays-pokemon/issues/6), child of the MVP architecture spec map (issue #1).

Source repo: https://github.com/davidhershey/ClaudePlaysPokemonStarter
Pinned to commit `2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8` (`main`, "Merge pull request #1 from MrCheeze/main", 2025-04-03). All links below point at this commit so line numbers stay stable.

## 1. Repo shape

The whole thing is 7 files, ~750 non-generated lines of Python:

- [`main.py`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/main.py) — CLI entrypoint (argparse: `--rom`, `--steps`, `--display`, `--sound`, `--max-history`, `--load-state`), constructs a `SimpleAgent` and calls `.run()`.
- [`config.py`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/config.py) — `MODEL_NAME = "claude-3-7-sonnet-20250219"`, `TEMPERATURE`, `MAX_TOKENS`, `USE_NAVIGATOR` flag.
- [`agent/emulator.py`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/emulator.py) — PyBoy wrapper: button presses, screenshots, pathfinding, collision map, state save/load.
- [`agent/memory_reader.py`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/memory_reader.py) — `PokemonRedReader`: raw RAM-address reads plus enums (species, moves, map locations, badges, items, status conditions).
- [`agent/simple_agent.py`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py) — the decision loop: builds Claude messages/tools, calls the Anthropic API, parses `tool_use` blocks, dispatches to the emulator, manages conversation-history summarization.
- `__init__.py`, `agent/__init__.py` — empty.
- [`README.md`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/README.md) — setup/usage docs, also states the 5-step "how it works" loop (screenshot → memory read → send to Claude → Claude responds with commands → execute).

## 2. Game / ROM target

**Confirmed: Pokemon Red.** `README.md` line 3: "A minimal implementation of Claude playing Pokemon Red using the PyBoy emulator." The default ROM filename is `pokemon.gb` ([`main.py:21`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/main.py#L21)), and every RAM address in `memory_reader.py` is annotated against Pokemon Red's known memory map (money at `0xD347-0xD349`, badges at `0xD356`, party base addresses starting `0xD16B`, etc. — these are the well-documented Pokemon Red/Blue (Gen 1) RAM offsets). No Pokemon Blue/Yellow branching logic exists; it's Red-specific (species table at [`memory_reader.py:85-277`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/memory_reader.py#L85-L277) and map-location table are keyed to Red's internal IDs).

## 3. Emulator

**PyBoy**, pinned to `pyboy==2.2.0` in [`requirements.txt:3`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/requirements.txt#L3). Instantiated in [`agent/emulator.py:14-27`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/emulator.py#L14-L27) as `Emulator.__init__`, with `window="null"` for headless mode or a real window + optional sound for display mode, `cgb=True` (Game Boy Color mode) in both cases.

## 4. Game-state extraction: RAM-based vs. vision-based

Both exist today, and they are cleanly separable:

- **RAM-based (fully decision-core-agnostic, reusable as-is):** `agent/memory_reader.py`'s `PokemonRedReader` reads player name/rival name (`0xD158`, `0xD34A`), money (BCD decode at `0xD347-49`), badges (bitflags at `0xD356`), location (`0xD35E`), tileset (`0xD367`), coordinates (`0xD361-62`), inventory (`0xD31D` count + `0xD31E` item table), dialog text (tilemap buffer scan `0xC3A0-0xC507`), party Pokemon (species/HP/level/types/moves/PP/status, base addresses `0xD16B` etc.), and Pokédex caught count (`0xD2F7-0xD30A` bitcount). `Emulator.get_state_from_memory()` ([`agent/emulator.py:488-537`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/emulator.py#L488-L537)) assembles all of this into one plain-text block. `Emulator.get_collision_map()` / `get_valid_moves()` / `find_path()` ([`agent/emulator.py:147-486`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/emulator.py#L147-L486)) derive an ASCII terrain map, legal-move list, and A* path purely from `pyboy.game_wrapper.game_area()` / `game_area_collision()` / sprite data — no pixels or vision model involved, just structured PyBoy state.
- **Vision-based (Claude-specific, not reusable for Jev as-is):** `Emulator.get_screenshot()` ([`agent/emulator.py:42-44`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/emulator.py#L42-L44)) returns a PIL image from `pyboy.screen.ndarray`. `simple_agent.py`'s `get_screenshot_base64()` ([`agent/simple_agent.py:17-27`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L17-L27)) upscales and base64-encodes it, and it is attached as an `{"type": "image", ...}` content block in every tool result ([`agent/simple_agent.py:150-158`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L150-L158), also in the summarization flow at [`agent/simple_agent.py:311-373`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L311-L373)). The model is expected to *look at* this image to understand the screen — that's a hard dependency on Claude's vision capability and has no equivalent for Jev (text-only). The RAM-based state (dialog, party, collision map, coordinates) already covers most of what the screenshot would otherwise convey, which is good news for a text-only swap-in.

## 5. Action space

Raw Game Boy button presses only, no macro/scripted actions beyond one optional convenience wrapper:

- **Core action primitive:** `Emulator.press_buttons(buttons, wait=True)` ([`agent/emulator.py:56-84`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/emulator.py#L56-L84)) takes a list of raw button names (`a`, `b`, `start`, `select`, `up`, `down`, `left`, `right`), presses/releases each with tick-based timing. This is exposed to Claude as the `press_buttons` tool ([`agent/simple_agent.py:50-73`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L50-L73)).
- **Optional macro action:** `navigate_to(row, col)` ([`agent/simple_agent.py:75-93`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L75-L93), gated by `config.USE_NAVIGATOR`, default `False`), which calls `Emulator.find_path()`'s A* implementation and replays the resulting direction sequence as individual button presses. It's a higher-level "go to grid cell" action built entirely on the RAM/collision data, no vision, no LLM judgment about pathing itself — only the decision of *which* cell to target is left to the model.
- No other action abstractions (menu shortcuts, battle-move-by-name, etc.) exist in this starter; everything funnels through raw buttons or the one grid-navigation macro.

## 6. Reusable as-is vs. must-replace for Jev

**Reusable as-is (no coupling to Claude/vision/free-text reasoning):**
- `agent/emulator.py` in full — PyBoy wrapper, button execution, RAM-derived collision map, valid-moves list, A* `find_path`, save-state load, `stop()`. All pure emulator/game-state mechanics.
- `agent/memory_reader.py` in full — `PokemonRedReader` and all its enums/dataclasses. Pure RAM parsing, zero LLM coupling.
- `Emulator.get_state_from_memory()` text assembly — could be reused directly or restructured into whatever typed shape Jev consumes (its raw RAM reads and string layout are decision-core agnostic).
- `main.py`'s CLI arg-parsing/ROM-loading skeleton is reusable in spirit (structure the equivalent for a Jev-driven agent), though it directly instantiates `SimpleAgent` and would need to instantiate whatever replaces it.

**Tightly coupled to Claude, must be replaced/rewritten:**
- `agent/simple_agent.py`'s core loop (`run()`, [`agent/simple_agent.py:218-305`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L218-L305)) — built around `anthropic.Anthropic().messages.create()`, Claude's `tools`/`tool_use` function-calling schema, and free-text assistant reasoning blocks logged before each action. Jev is a typed-judgment model (Choice/Score/Noul + confidence), not a free-text/tool-calling chat model, so this loop's request/response shape doesn't map over; it needs a new decision loop built around Jev's typed primitives (e.g., a `Choice` over the legal button/macro set, informed by the RAM-derived state).
- `SYSTEM_PROMPT` / `SUMMARY_PROMPT` ([`agent/simple_agent.py:30-47`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L30-L47)) — free-text prompt engineering assuming a reasoning LLM that explains itself; not meaningful for Jev's typed-judgment interface.
- `AVAILABLE_TOOLS` Claude tool-schema definitions ([`agent/simple_agent.py:50-93`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L50-L93)) — Anthropic-specific JSON tool schema; would become a typed choice/action space definition instead.
- `get_screenshot_base64()` and every screenshot-attachment call site ([`agent/simple_agent.py:17-27`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L17-L27), and its three usages in `process_tool_call`/`summarize_history`) — pure vision-input plumbing; drop entirely since Jev is text-only. (`Emulator.get_screenshot()` itself can stay for stream/display purposes, just not fed to the decision core.)
- `process_tool_call()`'s Claude `tool_result` dict shape ([`agent/simple_agent.py:116-216`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L116-L216)) — wraps results as Anthropic content blocks (text + image); needs to become whatever result/observation shape feeds back into Jev's next judgment call.
- `summarize_history()` ([`agent/simple_agent.py:307-382`](https://github.com/davidhershey/ClaudePlaysPokemonStarter/blob/2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8/agent/simple_agent.py#L307-L382)) — uses Claude to free-text-summarize the chat history; a Jev-based agent likely needs a different context/state-carry strategy since Jev isn't doing open-ended narrative summarization.
- `config.py`'s `MODEL_NAME`/`TEMPERATURE`/`MAX_TOKENS` — Anthropic API params, replace with Jev's config surface.

**Judgment call / partially reusable:**
- `navigate_to` macro action and `USE_NAVIGATOR` flag — the underlying `find_path` A* logic is reusable, but the tool-schema wrapper exposing it to Claude needs to become whatever "propose a target cell" mechanism Jev uses (e.g. a `Choice` over candidate destinations).

## 7. License

**No `LICENSE` file exists in the repository**, and `gh api repos/davidhershey/ClaudePlaysPokemonStarter` reports `"license": null`. Confirmed via direct repo file listing (`__init__.py`, `agent/__init__.py`, `agent/emulator.py`, `agent/memory_reader.py`, `agent/simple_agent.py`, `config.py`, `main.py`, `README.md`, `requirements.txt`, `.gitignore` — no `LICENSE`/`LICENSE.md`/`COPYING`) and via the GitHub repository API metadata.

**Implication:** with no license declared, the repo is "all rights reserved" by default under copyright law — there is no explicit grant to fork, modify, or redistribute derivative code. GitHub's public-repo terms permit viewing and forking the repo on GitHub itself, but do not grant a license to reuse the code outside GitHub's platform features. **Before building on this code for the MVP, get explicit permission from the author (David Hershey) or confirm a license was added upstream since this audit (commit `2ab0551016aba9af3ffa6fbb6e747d1ec33c08f8`, 2025-04-03) — this is a blocking legal question, not just a technical one.**

## 8. Summary answer to the ticket's question

- **Game/ROM:** Pokemon Red, confirmed (README + Red-specific RAM map).
- **Emulator:** PyBoy 2.2.0.
- **State extraction:** Mixed today — RAM reads (`memory_reader.py`) already cover player/party/location/dialog/collision/inventory, and are reusable as-is; screenshots are the only vision-dependent piece and are cleanly separable from the RAM path.
- **Action space:** Raw Game Boy button presses, plus one optional RAM-derived pathfinding macro (`navigate_to`); no other higher-level actions.
- **Reusable as-is:** `agent/emulator.py`, `agent/memory_reader.py` — the entire state-extraction and action-execution layer.
- **Must replace:** everything in `agent/simple_agent.py` that constructs Claude prompts/tool schemas, attaches screenshots, or drives the Anthropic messages API — this is where Jev's typed-judgment decision core needs to be substituted in.
- **License:** none declared (`license: null` via GitHub API, no LICENSE file in tree) — reuse permission is not yet established and should be confirmed with the author before building on this code.
