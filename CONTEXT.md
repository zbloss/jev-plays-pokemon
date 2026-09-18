# jev-plays-pokemon

A livestreamed demo where TypeSafe's Jev model plays Pokemon: Jev makes the tactical decisions, the rest of the system extracts game state and executes the chosen actions.

## Language

**Jev**:
TypeSafe's System One model: a fast (~150ms), text-only model that returns typed, constrained judgments (Choice / Score / Noul) with confidence, not free-form generated text or chain-of-thought reasoning.
_Avoid_: the AI, the model, the LLM (this project also touches an actual vision LLM — always say "Jev" when the tactical decision-maker is meant)

**System One**:
TypeSafe's product category for fast, structured-decision models, of which Jev is the flagship. Distinguishes this class of model from reasoning/"System Two" LLMs that generate free text.

**Choice / Score / Noul**:
The three typed judgment primitives Jev returns. Choice selects one of a defined set of options; Score gives a probability-weighted position on an ordered set of levels; Noul gives the probability that a yes/no condition holds. Every judgment is one of these three, always paired with a confidence.
_Avoid_: "answer", "output" (be specific about which primitive)

**Tactical action-selection loop**:
The repeating cycle of: read game state → ask Jev a Choice/Score/Noul question over the current legal actions → execute the chosen action. "Tactical" distinguishes this from long-horizon planning/strategy, which Jev does not do on its own.
_Avoid_: "the agent loop", "the game loop" (both undersell that Jev's role is bounded to one tactical decision at a time)

**Action space**:
The full set of options Jev may choose from in a tactical decision: the raw Game Boy button presses plus the navigation macro, offered together as one Choice's options. "Current legal actions" (see tactical action-selection loop) is the per-turn subset of this space that's actually selectable.
_Avoid_: "actions" alone when the full set, as opposed to one turn's legal subset, is meant

**Navigation macro**:
A single Choice option that walks Jev's character toward whatever target the current-objective milestone tracker is pointing at, using RAM-derived pathfinding executed over multiple emulator frames. Jev decides only whether to invoke it this turn; it takes no destination argument, so it stays a flat Choice option rather than a function-call.
_Avoid_: "navigate_to" (the reference starter's function name — not reused code, see #9's licensing decision)

**Game state**:
The structured text description of the current moment in the game that gets handed to Jev as input for a decision. Preferred source is direct extraction from emulator memory (RAM); a vision-model-as-tool-call is the fallback path when reliable RAM extraction isn't available for a given piece of state.
_Avoid_: "the screen", "the frame" (game state is text handed to Jev, not an image — the image only exists on the vision-fallback path)

**MVP**:
This map's actual destination: Jev making live, end-to-end tactical decisions in a form that can be livestreamed. Explicitly does not require completing the game.
_Avoid_: "the demo" alone (ambiguous with the livestream event itself — say "the MVP" for the system, "the stream" for the broadcast)

**Full completion**:
The follow-on, out-of-scope-for-this-map goal: a system capable of playing through the entire game, defeating all 8 Gym Leaders and finishing the storyline. Deliberately left as fog until the MVP reveals how Jev actually behaves — see `docs/adr/` once that decision is recorded.
_Avoid_: conflating with MVP scope
