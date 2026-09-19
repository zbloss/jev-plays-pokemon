from jev_plays_pokemon.milestones import MilestoneTarget, track_milestones

_EXPECTED_ORDER = (
    "got_starter",
    "got_oaks_parcel",
    "got_pokedex",
    "boulder_badge",
    "cascade_badge",
    "got_ss_ticket",
    "thunder_badge",
    "rainbow_badge",
    "got_poke_flute",
    "soul_badge",
    "marsh_badge",
    "volcano_badge",
    "earth_badge",
    "beat_champion",
)

# Deliberately re-declared rather than imported from `milestones` (its flag
# constants are private): keeps these fixtures asserting against literal
# values sourced independently from pokered, so a future edit to the
# production constants can't silently drag a stale test along with it.
_EVENT_GOT_STARTER = 34
_EVENT_GOT_OAKS_PARCEL = 57
_EVENT_GOT_POKEDEX = 37
_EVENT_GOT_POKE_FLUTE = 296
_EVENT_GOT_SS_TICKET = 1372
_EVENT_BEAT_CHAMPION_RIVAL = 2305


def test_fresh_save_has_no_completed_milestones_and_starts_at_the_first():
    progress = track_milestones(event_flags=frozenset(), badges=())

    assert progress.completed == ()
    assert progress.current is not None
    assert progress.current.milestone_id == "got_starter"
    assert [m.milestone_id for m in progress.future] == list(_EXPECTED_ORDER[1:])


def test_scripted_list_is_in_the_expected_story_order():
    progress = track_milestones(event_flags=frozenset(), badges=())

    assert progress.current is not None
    ids_in_order = [progress.current.milestone_id] + [
        m.milestone_id for m in progress.future
    ]
    assert tuple(ids_in_order) == _EXPECTED_ORDER


def test_completing_every_milestone_leaves_no_current_or_future():
    all_badges = (
        "BOULDERBADGE",
        "CASCADEBADGE",
        "THUNDERBADGE",
        "RAINBOWBADGE",
        "SOULBADGE",
        "MARSHBADGE",
        "VOLCANOBADGE",
        "EARTHBADGE",
    )
    all_flags = frozenset(
        {
            _EVENT_GOT_STARTER,
            _EVENT_GOT_OAKS_PARCEL,
            _EVENT_GOT_POKEDEX,
            _EVENT_GOT_POKE_FLUTE,
            _EVENT_GOT_SS_TICKET,
            _EVENT_BEAT_CHAMPION_RIVAL,
        }
    )

    progress = track_milestones(event_flags=all_flags, badges=all_badges)

    assert [m.milestone_id for m in progress.completed] == list(_EXPECTED_ORDER)
    assert progress.current is None
    assert progress.future == ()


def test_split_reflects_partial_progress_through_the_scripted_list():
    event_flags = frozenset(
        {_EVENT_GOT_STARTER, _EVENT_GOT_OAKS_PARCEL, _EVENT_GOT_POKEDEX}
    )
    badges = ("BOULDERBADGE", "CASCADEBADGE")

    progress = track_milestones(event_flags=event_flags, badges=badges)

    assert [m.milestone_id for m in progress.completed] == [
        "got_starter",
        "got_oaks_parcel",
        "got_pokedex",
        "boulder_badge",
        "cascade_badge",
    ]
    assert progress.current is not None
    assert progress.current.milestone_id == "got_ss_ticket"
    assert [m.milestone_id for m in progress.future] == [
        "thunder_badge",
        "rainbow_badge",
        "got_poke_flute",
        "soul_badge",
        "marsh_badge",
        "volcano_badge",
        "earth_badge",
        "beat_champion",
    ]


def test_a_later_flag_completed_out_of_order_does_not_skip_the_earlier_current_milestone():
    # Only the champion flag is set; every earlier milestone is still unmet,
    # so the tracker must not report "beat_champion" as completed or current.
    progress = track_milestones(
        event_flags=frozenset({_EVENT_BEAT_CHAMPION_RIVAL}), badges=()
    )

    assert progress.completed == ()
    assert progress.current is not None
    assert progress.current.milestone_id == "got_starter"


def test_unrelated_event_flags_and_badges_are_ignored():
    progress = track_milestones(
        event_flags=frozenset({0, 1, 999999}), badges=("TOWN MAP",)
    )

    assert progress.completed == ()
    assert progress.current is not None
    assert progress.current.milestone_id == "got_starter"


def test_current_milestone_exposes_a_map_target_for_the_navigation_macro():
    progress = track_milestones(event_flags=frozenset(), badges=())

    assert progress.current is not None
    target = progress.current.target
    assert target.map_id == 40
    assert target.map_name == "Oaks Lab"


def test_target_map_advances_with_progress():
    event_flags = frozenset({_EVENT_GOT_STARTER, _EVENT_GOT_OAKS_PARCEL})
    progress = track_milestones(event_flags=event_flags, badges=())

    assert progress.current is not None
    assert progress.current.milestone_id == "got_pokedex"
    assert progress.current.target.map_name == "Oaks Lab"


def test_all_scripted_milestones_have_a_resolved_tile_target():
    # All 14 scripted milestones carry a ROM-derived tile-level destination
    # (issue #98) - `target_x`/`target_y` are only ever `None` for a future
    # milestone added without one yet (see `MilestoneTarget`'s docstring).
    progress = track_milestones(event_flags=frozenset(), badges=())
    all_milestones = (
        list(progress.completed) + [progress.current] + list(progress.future)
    )

    assert len(all_milestones) == len(_EXPECTED_ORDER)
    for milestone in all_milestones:
        assert milestone is not None
        assert milestone.target.target_x is not None, milestone.milestone_id
        assert milestone.target.target_y is not None, milestone.milestone_id


def test_milestone_target_can_carry_a_verified_tile_destination():
    target = MilestoneTarget(map_id=40, map_name="Oaks Lab", target_x=4, target_y=5)

    assert target.target_x == 4
    assert target.target_y == 5
