# A repeatable method for ROM-verifying tile-level milestone targets

Resolves issue [#90](https://github.com/zbloss/jev-plays-pokemon/issues/90), child of the wayfinder-map
issue [#85](https://github.com/zbloss/jev-plays-pokemon/issues/85), the second half of that map's
destination (the first half - the cross-map travel architecture - is `docs/adr/0002-travel-graph-for-cross-map-navigation.md`).

## Question

What is the documented, repeatable method for producing a ROM-verified tile-level `target_x`/`target_y`
for a scripted milestone (`milestones.py`'s `MilestoneTarget`), and can it be demonstrated end-to-end
against at least one real milestone?

## The method

1. **Boot to a controllable state.** `emulator.boot_to_controllable_state()` (or, in a test, `boot_past_intro`
   against a fresh `PyBoy(rom_path, window="null")`) mashes through Pokemon Red's un-skippable intro to a
   controllable, real game state in the player's bedroom - this is already a solved, reused problem
   (`emulator.py`, `tests/test_navigation.py`'s `bedroom_state` fixture); don't re-derive it.
2. **Drive to the target position with `execute_button`, one tile at a time, by watching the screen.**
   There's no shortcut for "where is this NPC/object in world-tile terms" other than actually walking there
   and looking: run a short script that, after each `execute_button` call, saves `pyboy.screen.image` (a
   `PIL.Image`, works fine headless under `window="null"`) to a file and reads it back to see where the
   player actually ended up on screen. This is the same discipline `emulator.py`'s docstring already
   documents `_HOUSE_EXIT_PATH`/`boot_past_intro` were built with ("lifted verbatim from the one
   `tests/test_navigation.py` worked out by booting this repo's own `pokemon_red.gb` headless and
   inspecting rendered frames at each step") - this ticket did the same thing again, this time walking from
   the outdoor Pallet Town spawn into Oak's Lab.
   - `pyboy.save_state()`/`load_state()` checkpoints while exploring let a wrong turn be retried from the
     last good spot instead of re-walking the whole route on every attempt.
   - Once a working route is found by trial and error, **re-run the whole thing as one continuous script
     from a cold boot** (not stitched together from saved mid-states) to confirm it's actually deterministic
     end to end before writing it down as *the* method for that target - see the "post-warp jump" gotcha
     below for why this matters.
3. **Read the position via the same three addresses `game_state.py` already uses**, not new ones:
   `memory[0xD35E]` (`map_id`), `memory[0xD362]` (`player_x`), `memory[0xD361]` (`player_y`) - exactly
   `tests/test_navigation.py`'s existing `_position` helper.
4. **Confirm the tile is the actual interaction trigger, not just visually adjacent to the target NPC/object.**
   Standing next to something in Gen 1 does nothing; the player has to be on the specific tile the object's
   script checks, facing it, and press "a". Verify by pressing "a" and checking `GameState.dialog_open`
   (`game_state.py`'s tilemap-derived dialog-arrow flag) goes `True` over the following ~1-2 seconds of
   ticks - the arrow only appears once the game has printed enough text to need a "press to continue"
   glyph, so check across several `tick(30, True)` chunks rather than immediately after the button press.
5. **Record**: the exact button sequence from the nearest reusable checkpoint (bedroom, or an existing
   `*_state` fixture), the resulting `(map_id, player_x, player_y)`, and how it was confirmed interactive.

## Gotcha: a position read immediately after a map transition can look like it moved diagonally

Immediately after a map transition (walking out of a building, or into one), the *next* `execute_button`
call can report a position change that doesn't match the direction pressed - e.g. pressing "down" once,
immediately after exiting the player's house, moved the read position by `(+2, -2)` tiles instead of the
expected `(0, +1)`. This isn't a bug in `execute_button` (see its own docstring: it holds a direction until
the player's RAM position actually changes, capped at `_MAX_WALK_FRAMES`) - the position genuinely settles
at that new value and stays there. The likely cause: the game engine runs its own automatic post-warp
positioning over the first several frames after a transition (independent of player input), and the first
`execute_button` call after a warp can return once *that* finishes moving the RAM position, rather than
once the player's own subsequent press has been fully applied on top of it.

Practical consequence: don't treat the coordinate delta right after a warp as "one input's worth" of
movement, and don't trust a screenshot saved in the same frame a state was loaded/warped into - PyBoy only
refreshes `pyboy.screen.image`/`.ndarray` on a `render=True` tick, so a screenshot taken immediately after
`load_state` (before any further tick) can still show the *previous* map's stale tile buffer even though
`pyboy.memory[...]` already reads the new position correctly (observed directly while building this
ticket's worked example below - see `game_state.py`'s and `navigation.py`'s docstrings for the same
render=True-vs-stale-buffer distinction elsewhere in this codebase). Always confirm a post-warp read against
a screenshot taken after at least one further `render=True` tick, not the frame the warp itself landed on.

## Worked example: `got_starter`'s target

`milestones.py`'s first milestone, `got_starter`, targets Oak's Lab (`map_id` 40) but - like every other
milestone today - carries no tile-level coordinates (`MilestoneTarget.target_x`/`target_y` both `None`).

Following the method above: from the outdoor Pallet Town spawn (`(map_id=0, x=3, y=7)`, reached by
`tests/test_navigation.py`'s existing `_HOUSE_EXIT_PATH`), a fixed 28-button sequence
(`tests/test_navigation.py`'s `_TO_OAKS_LAB_TABLE_PATH`) walks the player south and east to Oak's Lab's
front door (the door itself sits at Pallet Town tile `(12, 12)`), then north through the lab's front room to
the starter Poke Ball table, landing at:

```
map_id = 40 (Oak's Lab), player_x = 7, player_y = 4
```

Confirmed as the real interaction tile, not just an adjacent one: pressing "a" from this position opens a
dialog box (`GameState.dialog_open` goes `True` within the following ~120 frames - the game's "There are 3
Pokemon here!" line). Re-verified end to end from a cold boot (not stitched from saved mid-states) to
confirm determinism, and captured as a real, ROM-gated test:
[`test_walking_to_oaks_lab_starter_table_reaches_a_rom_verified_tile`](../../tests/test_navigation.py)
(runs for real against PyBoy + `pokemon_red.gb` wherever the gitignored ROM is present, same as every
other test in that file).

**Caveat**: reaching this exact spot in a real playthrough normally happens automatically - Oak intercepts
the player in Route 1's tall grass and walks them into the lab, rather than the player choosing to walk in
through the front door themselves. The sequence above enters manually instead, since the point here is a
reproducible RAM-verification method, not reproducing that specific scripted encounter. The recorded
`(target_x=7, target_y=4)` is still a legitimate, ROM-verified tile-level destination for the milestone
regardless of how the player arrives there (matches what `MilestoneTarget.target_x`/`target_y` mean per
`milestones.py`'s own docstring) - wiring it into `milestones.py`'s `got_starter` entry, and ROM-verifying
the remaining 13 milestones' targets the same way, is ordinary [#84](https://github.com/zbloss/jev-plays-pokemon/issues/84)
follow-up work, not done here (per [#85](https://github.com/zbloss/jev-plays-pokemon/issues/85)'s Notes:
this map is decision-only).
