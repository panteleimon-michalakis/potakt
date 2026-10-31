"""Per-template semantics of the checker (ADR-0005 §4): zero, one and several violations."""

from typing import Any

import pytest
from checker_helpers import DST_WEEK, WEEK, make_dm, outcome, res, rule, run, task

from potakt.model import DecisionModel, Task

R1 = res("r1", skills=["als"], contract="full_time", level=2)
R2 = res("r2", skills=[], contract="part_time", level=1)
R3 = res("r3", skills=["als", "opening"], contract="full_time")
RESOURCES = [R1, R2, R3]

# Monday 2 Nov 2026 to Wednesday 4 Nov, Europe/Athens.
MON_AM = task("mon-am", "2026-11-02T07:00", "2026-11-02T15:00", demand=2, type="am")
MON_PM = task("mon-pm", "2026-11-02T15:00", "2026-11-02T23:00", type="pm")
MON_N = task("mon-n", "2026-11-02T23:00", "2026-11-03T07:00", type="night", required_skill="als")
TUE_AM = task("tue-am", "2026-11-03T07:00", "2026-11-03T15:00", demand=2, type="am")
TUE_N = task("tue-n", "2026-11-03T23:00", "2026-11-04T07:00", type="night", required_skill="als")
TASKS = [MON_AM, MON_PM, MON_N, TUE_AM, TUE_N]


def one_rule(*rules: dict[str, Any], horizon: tuple[str, str] = WEEK) -> DecisionModel:
    """A DM whose coverage only looks at a task id no test uses, plus ``rules``."""
    coverage = rule("coverage", scope={"tasks": [{"attribute": "id", "op": "eq", "value": "-"}]})
    return make_dm(*rules, coverage=coverage, horizon=horizon)


# --- coverage --------------------------------------------------------------------------------


def test_coverage_is_zero_when_every_task_has_its_demand() -> None:
    dm = make_dm()
    assignments = {
        "mon-am": ["r1", "r2"],
        "mon-pm": ["r3"],
        "mon-n": ["r1"],
        "tue-am": ["r2", "r3"],
        "tue-n": ["r3"],
    }

    result = run(dm, RESOURCES, TASKS, assignments).rule("coverage")

    assert result.amount == 0
    assert result.violations == ()


def test_coverage_counts_missing_and_surplus_resources_per_task() -> None:
    dm = make_dm()
    assignments = {"mon-am": ["r1"], "mon-pm": ["r1", "r2", "r3"], "mon-n": ["r3"]}

    result = run(dm, RESOURCES, TASKS, assignments).rule("coverage")

    assert outcome(result) == [
        (None, ("mon-am",), None, 1),
        (None, ("mon-pm",), None, 2),
        (None, ("tue-am",), None, 2),
        (None, ("tue-n",), None, 1),
    ]
    assert result.amount == 6
    assert result.violations[0].message == "task mon-am: 1 assigned, demand 2"


def test_coverage_only_counts_in_scope_tasks_and_resources() -> None:
    coverage = rule(
        "coverage",
        scope={
            "resources": [{"attribute": "skills", "op": "contains", "value": "als"}],
            "tasks": [{"attribute": "type", "op": "eq", "value": "am"}],
        },
    )
    dm = make_dm(coverage=coverage)

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "tue-am": ["r1", "r3"]})

    assert outcome(result.rule("coverage")) == [(None, ("mon-am",), None, 1)]


# --- requires_attribute ----------------------------------------------------------------------


def requires(match: str = "contains", **fields: Any) -> dict[str, Any]:
    params = {"task_attribute": "required_skill", "resource_attribute": "skills", "match": match}
    return rule("requires_attribute", params=params, **fields)


def test_requires_attribute_is_zero_for_matching_resources_and_unrestricted_tasks() -> None:
    dm = one_rule(requires())

    result = run(dm, RESOURCES, TASKS, {"mon-n": ["r1"], "tue-n": ["r3"], "mon-am": ["r2"]})

    assert result.rule("requires_attribute").violations == ()


def test_requires_attribute_reports_each_non_matching_assignment() -> None:
    dm = one_rule(requires())

    one = run(dm, RESOURCES, TASKS, {"mon-n": ["r2"]}).rule("requires_attribute")
    several = run(dm, RESOURCES, TASKS, {"mon-n": ["r2", "r1"], "tue-n": ["r2"]})

    assert outcome(one) == [("r2", ("mon-n",), None, 1)]
    assert one.violations[0].message == (
        'r2 on mon-n: skills does not contain required_skill "als"'
    )
    assert outcome(several.rule("requires_attribute")) == [
        ("r2", ("mon-n",), None, 1),
        ("r2", ("tue-n",), None, 1),
    ]


def test_requires_attribute_contains_needs_every_listed_item() -> None:
    both = task("both", "2026-11-02T07:00", "2026-11-02T15:00", needs=["als", "opening"])
    params = {"task_attribute": "needs", "resource_attribute": "skills", "match": "contains"}
    dm = one_rule(rule("requires_attribute", params=params))

    result = run(dm, RESOURCES, [both], {"both": ["r1", "r3"]})

    assert outcome(result.rule("requires_attribute")) == [("r1", ("both",), None, 1)]


def test_requires_attribute_equals_compares_values_and_needs_the_attribute() -> None:
    senior = task("senior", "2026-11-02T07:00", "2026-11-02T15:00", min_level=2)
    params = {"task_attribute": "min_level", "resource_attribute": "level", "match": "equals"}
    dm = one_rule(rule("requires_attribute", params=params))

    result = run(dm, RESOURCES, [senior], {"senior": ["r1", "r2", "r3"]})

    assert outcome(result.rule("requires_attribute")) == [
        ("r2", ("senior",), None, 1),
        ("r3", ("senior",), None, 1),
    ]
    assert result.rule("requires_attribute").violations[0].message == (
        "r2 on senior: level does not equal min_level 2"
    )


def test_requires_attribute_equals_ignores_list_order_and_matches_scalars() -> None:
    pair = task("pair", "2026-11-02T07:00", "2026-11-02T15:00", needs=["opening", "als"])
    solo = task("solo", "2026-11-03T07:00", "2026-11-03T15:00", needs="als")
    params = {"task_attribute": "needs", "resource_attribute": "skills", "match": "equals"}
    dm = one_rule(rule("requires_attribute", params=params))

    result = run(dm, RESOURCES, [pair, solo], {"pair": ["r1", "r3"], "solo": ["r1", "r3"]})

    assert outcome(result.rule("requires_attribute")) == [
        ("r1", ("pair",), None, 1),
        ("r3", ("solo",), None, 1),
    ]


# --- no_overlap ------------------------------------------------------------------------------


def test_touching_tasks_do_not_overlap() -> None:
    dm = one_rule(rule("no_overlap"))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1"], "mon-pm": ["r1"], "mon-n": ["r1"]})

    assert result.rule("no_overlap").violations == ()


def test_no_overlap_reports_each_overlapping_pair_once() -> None:
    long_day = task("long", "2026-11-02T10:00", "2026-11-03T08:00")
    dm = one_rule(rule("no_overlap"))

    one = run(dm, RESOURCES, [*TASKS, long_day], {"mon-am": ["r1"], "long": ["r1"]})
    several = run(
        dm,
        RESOURCES,
        [*TASKS, long_day],
        {"long": ["r1", "r2"], "mon-am": ["r1"], "mon-pm": ["r1"], "tue-am": ["r1", "r2"]},
    )

    assert outcome(one.rule("no_overlap")) == [("r1", ("long", "mon-am"), None, 1)]
    assert one.rule("no_overlap").violations[0].message == "r1: long overlaps mon-am"
    assert outcome(several.rule("no_overlap")) == [
        ("r1", ("long", "mon-am"), None, 1),
        ("r1", ("long", "mon-pm"), None, 1),
        ("r1", ("long", "tue-am"), None, 1),
        ("r2", ("long", "tue-am"), None, 1),
    ]


def test_no_overlap_ignores_out_of_scope_tasks() -> None:
    long_day = task("long", "2026-11-02T10:00", "2026-11-03T08:00", type="training")
    scope = {"tasks": [{"attribute": "type", "op": "ne", "value": "training"}]}
    dm = one_rule(rule("no_overlap", scope=scope))

    result = run(dm, RESOURCES, [*TASKS, long_day], {"mon-am": ["r1"], "long": ["r1"]})

    assert result.rule("no_overlap").violations == ()


# --- min_rest --------------------------------------------------------------------------------


def rest_rule(hours: float = 11) -> dict[str, Any]:
    return rule("min_rest", params={"hours": hours})


def test_rest_of_exactly_the_minimum_is_allowed_and_one_minute_less_is_not() -> None:
    late = task("late", "2026-11-02T15:00", "2026-11-02T23:00")
    on_time = task("on-time", "2026-11-03T10:00", "2026-11-03T18:00")
    early = task("early", "2026-11-03T09:59", "2026-11-03T17:59")
    dm = one_rule(rest_rule())

    ok = run(dm, RESOURCES, [late, on_time], {"late": ["r1"], "on-time": ["r1"]})
    short = run(dm, RESOURCES, [late, early], {"late": ["r1"], "early": ["r1"]})

    assert ok.rule("min_rest").violations == ()
    assert outcome(short.rule("min_rest")) == [("r1", ("early", "late"), None, 1)]
    assert short.rule("min_rest").violations[0].message == (
        "r1: 10 h 59 min rest between late and early, minimum 11 h"
    )


def test_min_rest_counts_every_ordered_pair_within_the_gap() -> None:
    dm = one_rule(rest_rule(16))
    assignments = {"mon-am": ["r1"], "mon-pm": ["r1"], "mon-n": ["r1"], "tue-am": ["r2"]}

    result = run(dm, RESOURCES, TASKS, assignments).rule("min_rest")

    # am→pm 0 h, am→night 8 h, pm→night 0 h; every pair is under 16 h.
    assert outcome(result) == [
        ("r1", ("mon-am", "mon-n"), None, 1),
        ("r1", ("mon-am", "mon-pm"), None, 1),
        ("r1", ("mon-n", "mon-pm"), None, 1),
    ]


def test_min_rest_leaves_overlapping_pairs_to_no_overlap() -> None:
    overlapping = task("overlap", "2026-11-02T14:00", "2026-11-02T16:00")
    dm = one_rule(rest_rule())

    result = run(dm, RESOURCES, [MON_AM, overlapping], {"mon-am": ["r1"], "overlap": ["r1"]})

    assert result.rule("min_rest").violations == ()


def test_min_rest_uses_real_minutes_across_the_dst_fall_back() -> None:
    # Clocks go back 04:00 -> 03:00 on 25 Oct: 20:00 to 06:00 local is 11 real hours.
    evening = task("evening", "2026-10-24T12:00", "2026-10-24T20:00")
    morning = task("morning", "2026-10-25T06:00", "2026-10-25T14:00")
    dm = one_rule(rest_rule(), horizon=DST_WEEK)

    result = run(dm, RESOURCES, [evening, morning], {"evening": ["r1"], "morning": ["r1"]})

    assert result.rule("min_rest").violations == ()


# --- max_per_period --------------------------------------------------------------------------


def limit_rule(template: str, measure: str, period: str, limit: float, **fields: Any):
    params = {"measure": measure, "period": period, "limit": limit}
    return rule(template, params=params, **fields)


def test_max_count_per_day_is_zero_within_the_limit() -> None:
    dm = one_rule(limit_rule("max_per_period", "count", "day", 1))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1"], "mon-n": ["r2"], "tue-am": ["r1"]})

    assert result.rule("max_per_period").violations == ()


def test_max_count_per_day_counts_tasks_on_their_start_day() -> None:
    dm = one_rule(limit_rule("max_per_period", "count", "day", 1))
    assignments = {"mon-am": ["r1", "r2"], "mon-pm": ["r1"], "mon-n": ["r1", "r2"]}

    result = run(dm, RESOURCES, TASKS, assignments).rule("max_per_period")

    assert outcome(result) == [
        ("r1", ("mon-am", "mon-n", "mon-pm"), "2026-11-02", 2),
        ("r2", ("mon-am", "mon-n"), "2026-11-02", 1),
    ]
    assert result.violations[0].message == "r1 in 2026-11-02: 3 tasks, limit 1"


def test_max_count_per_week_and_horizon_use_their_labels() -> None:
    week = limit_rule("max_per_period", "count", "week", 1, id="week")
    horizon = limit_rule("max_per_period", "count", "horizon", 1, id="horizon")
    dm = one_rule(week, horizon)

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1"], "tue-am": ["r1"]})

    assert outcome(result.rule("week")) == [("r1", ("mon-am", "tue-am"), "2026-W45", 1)]
    assert outcome(result.rule("horizon")) == [
        ("r1", ("mon-am", "tue-am"), "2026-11-02/2026-11-08", 1)
    ]


def test_max_hours_counts_started_hours_of_excess() -> None:
    dm = one_rule(limit_rule("max_per_period", "hours", "week", 15.5))
    within = run(dm, RESOURCES, TASKS, {"mon-am": ["r1"]})
    half_hour = run(dm, RESOURCES, TASKS, {"mon-am": ["r1"], "mon-pm": ["r1"]})
    three_tasks = run(dm, RESOURCES, TASKS, {"mon-am": ["r1"], "mon-pm": ["r1"], "tue-am": ["r1"]})

    assert within.rule("max_per_period").violations == ()
    assert outcome(half_hour.rule("max_per_period")) == [
        ("r1", ("mon-am", "mon-pm"), "2026-W45", 1)
    ]
    assert half_hour.rule("max_per_period").violations[0].message == (
        "r1 in 2026-W45: 16 h worked, limit 15 h 30 min"
    )
    # 24 h worked, 8 h 30 min over the limit: 9 started hours.
    assert three_tasks.rule("max_per_period").amount == 9


def test_night_across_the_dst_fall_back_counts_its_real_length() -> None:
    night = task("night", "2026-10-24T23:00", "2026-10-25T07:00")
    eight = one_rule(limit_rule("max_per_period", "hours", "day", 8), horizon=DST_WEEK)
    nine = one_rule(limit_rule("max_per_period", "hours", "day", 9), horizon=DST_WEEK)

    over = run(eight, RESOURCES, [night], {"night": ["r1"]}).rule("max_per_period")
    exact = run(nine, RESOURCES, [night], {"night": ["r1"]}).rule("max_per_period")

    assert outcome(over) == [("r1", ("night",), "2026-10-24", 1)]
    assert over.violations[0].message == "r1 in 2026-10-24: 9 h worked, limit 8 h"
    assert exact.violations == ()


# --- min_per_period --------------------------------------------------------------------------


def test_min_per_period_is_zero_when_every_period_reaches_the_limit() -> None:
    scope = {"resources": [{"attribute": "id", "op": "eq", "value": "r1"}]}
    rule_doc = limit_rule("min_per_period", "count", "day", 1, scope=scope)
    dm = one_rule(rule_doc, horizon=("2026-11-02", "2026-11-03"))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1"], "tue-am": ["r1"]})

    assert result.rule("min_per_period").violations == ()


def test_min_per_period_reports_periods_without_any_work() -> None:
    dm = one_rule(
        limit_rule("min_per_period", "count", "day", 1), horizon=("2026-11-02", "2026-11-04")
    )

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "tue-am": ["r1"]})

    assert outcome(result.rule("min_per_period")) == [
        ("r1", (), "2026-11-04", 1),
        ("r2", (), "2026-11-03", 1),
        ("r2", (), "2026-11-04", 1),
        ("r3", (), "2026-11-02", 1),
        ("r3", (), "2026-11-03", 1),
        ("r3", (), "2026-11-04", 1),
    ]
    assert result.rule("min_per_period").violations[0].message == (
        "r1 in 2026-11-04: 0 tasks, minimum 1"
    )


def test_min_hours_counts_started_hours_of_shortfall() -> None:
    scope = {"resources": [{"attribute": "contract", "op": "eq", "value": "full_time"}]}
    dm = one_rule(limit_rule("min_per_period", "hours", "week", 16.5, scope=scope))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "mon-pm": ["r1"]})

    assert outcome(result.rule("min_per_period")) == [
        ("r1", ("mon-am", "mon-pm"), "2026-W45", 1),
        ("r3", (), "2026-W45", 17),
    ]
    assert result.rule("min_per_period").violations[0].message == (
        "r1 in 2026-W45: 16 h worked, minimum 16 h 30 min"
    )


# --- max_consecutive_days --------------------------------------------------------------------


def daily(prefix: str, days: range) -> list[Task]:
    return [task(f"{prefix}{d}", f"2026-11-{d:02d}T07:00", f"2026-11-{d:02d}T15:00") for d in days]


def test_consecutive_days_within_the_limit_are_fine() -> None:
    dm = one_rule(rule("max_consecutive_days", params={"days": 3}))
    shifts = daily("d", range(2, 9))

    result = run(dm, RESOURCES, shifts, {"d2": ["r1"], "d3": ["r1"], "d4": ["r1"]})

    assert result.rule("max_consecutive_days").violations == ()


def test_runs_split_by_one_free_day_are_scored_separately() -> None:
    dm = one_rule(rule("max_consecutive_days", params={"days": 2}))
    shifts = daily("d", range(2, 9))
    worked = ["d2", "d3", "d5", "d6", "d7"]  # 4 Nov is free

    result = run(dm, RESOURCES, shifts, {t: ["r1"] for t in worked})

    assert outcome(result.rule("max_consecutive_days")) == [
        ("r1", ("d5", "d6", "d7"), "2026-11-05/2026-11-07", 1)
    ]
    assert result.rule("max_consecutive_days").violations[0].message == (
        "r1: 3 consecutive working days 2026-11-05 to 2026-11-07, limit 2"
    )


def test_each_long_run_scores_its_length_over_the_limit() -> None:
    dm = one_rule(rule("max_consecutive_days", params={"days": 2}))
    shifts = [*daily("d", range(2, 9)), MON_N]
    assignments = {**{f"d{d}": ["r1"] for d in range(2, 9)}, "mon-n": ["r2"]}
    assignments["d3"] = ["r1", "r2"]
    assignments["d4"] = ["r1", "r2"]

    result = run(dm, RESOURCES, shifts, assignments)

    # r1 works all 7 days; r2 works the 2nd (night counts on its start day), 3rd and 4th.
    assert outcome(result.rule("max_consecutive_days")) == [
        ("r1", tuple(f"d{d}" for d in range(2, 9)), "2026-11-02/2026-11-08", 5),
        ("r2", ("d3", "d4", "mon-n"), "2026-11-02/2026-11-04", 1),
    ]


# --- unavailable -----------------------------------------------------------------------------


def unavailable(*windows: tuple[str, str, str], **fields: Any) -> dict[str, Any]:
    params = {"windows": [{"resource_id": r, "start": s, "end": e} for r, s, e in windows]}
    return rule("unavailable", params=params, **fields)


def test_window_starting_when_a_task_ends_is_not_violated() -> None:
    dm = one_rule(unavailable(("r1", "2026-11-02T15:00", "2026-11-02T23:00")))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1"], "mon-n": ["r1"]})

    assert result.rule("unavailable").violations == ()


def test_unavailable_counts_a_task_once_even_if_several_windows_overlap() -> None:
    dm = one_rule(
        unavailable(
            ("r1", "2026-11-02T14:00", "2026-11-02T16:00"),
            ("r1", "2026-11-02T22:00", "2026-11-03T08:00"),
            ("r2", "2026-11-02T00:00", "2026-11-03T00:00"),
        )
    )

    one = run(dm, RESOURCES, TASKS, {"mon-pm": ["r1"]}).rule("unavailable")
    several = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "mon-n": ["r1", "r3"]})

    assert outcome(one) == [("r1", ("mon-pm",), None, 1)]
    assert one.violations[0].message == (
        "r1 on mon-pm: overlaps unavailability 2026-11-02T14:00 to 2026-11-02T16:00"
    )
    assert outcome(several.rule("unavailable")) == [
        ("r1", ("mon-am",), None, 1),
        ("r1", ("mon-n",), None, 1),
        ("r2", ("mon-am",), None, 1),
    ]


# --- preference ------------------------------------------------------------------------------


def prefers(kind: str, weight: int = 3, **scope: Any) -> dict[str, Any]:
    return rule("preference", "soft", weight, params={"kind": kind}, scope=scope)


def test_avoid_preference_is_zero_without_matching_assignments() -> None:
    nights = [{"attribute": "type", "op": "eq", "value": "night"}]
    dm = one_rule(prefers("avoid", tasks=nights))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1"], "tue-am": ["r3"]})

    assert result.rule("preference").violations == ()
    assert result.score.soft == 0


def test_avoid_preference_penalizes_each_scoped_assignment() -> None:
    nights = [{"attribute": "type", "op": "eq", "value": "night"}]
    r1 = [{"attribute": "id", "op": "eq", "value": "r1"}]
    dm = one_rule(prefers("avoid", resources=r1, tasks=nights))

    one = run(dm, RESOURCES, TASKS, {"mon-n": ["r1"], "tue-am": ["r1"]})
    several = run(dm, RESOURCES, TASKS, {"mon-n": ["r1"], "tue-n": ["r1", "r3"]})

    assert outcome(one.rule("preference")) == [("r1", ("mon-n",), None, 1)]
    assert one.rule("preference").violations[0].message == "r1 on mon-n: avoided assignment"
    assert one.score.soft == 3
    assert several.rule("preference").amount == 2
    assert several.score.soft == 6


def test_prefer_preference_makes_soft_negative() -> None:
    mornings = [{"attribute": "type", "op": "eq", "value": "am"}]
    dm = one_rule(prefers("prefer", weight=2, tasks=mornings))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "tue-am": ["r1"]})

    preference = result.rule("preference")
    assert preference.amount == 3
    assert preference.contribution == -6
    assert preference.violations[0].message == "r1 on mon-am: preferred assignment"
    assert result.score.soft == -6
    assert result.score.hard == 0


# --- balance ---------------------------------------------------------------------------------


def balance(measure: str = "count", **scope: Any) -> dict[str, Any]:
    return rule("balance", "soft", 2, params={"measure": measure}, scope=scope)


def test_balance_is_zero_when_everyone_has_the_same_amount() -> None:
    dm = one_rule(balance())

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "mon-pm": ["r3"]})

    assert result.rule("balance").violations == ()


def test_balance_includes_resources_with_zero() -> None:
    dm = one_rule(balance())

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "tue-am": ["r1"]})

    assert outcome(result.rule("balance")) == [(None, (), None, 2)]
    assert result.rule("balance").violations[0].message == (
        "tasks per resource range from 0 (r3) to 2 (r1)"
    )
    assert result.score.soft == 4


def test_balance_hours_rounds_the_spread_up_to_started_hours() -> None:
    short = task("short", "2026-11-02T07:00", "2026-11-02T13:30", type="am")
    nights = [{"attribute": "type", "op": "ne", "value": "night"}]
    als = [{"attribute": "skills", "op": "contains", "value": "als"}]
    dm = one_rule(balance("hours", resources=als, tasks=nights))

    result = run(dm, RESOURCES, [MON_AM, short, MON_N], {"mon-am": ["r1"], "short": ["r3"]})

    assert outcome(result.rule("balance")) == [(None, (), None, 2)]
    assert result.rule("balance").violations[0].message == (
        "hours per resource range from 6 h 30 min (r3) to 8 h (r1)"
    )


# --- together / apart ------------------------------------------------------------------------


def pair_rule(template: str, hardness: str = "soft", **fields: Any) -> dict[str, Any]:
    return rule(template, hardness, params={"resource_ids": ["r1", "r2"]}, **fields)


def test_together_is_zero_when_the_pair_always_works_together() -> None:
    dm = one_rule(pair_rule("together"))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "mon-pm": ["r3"]})

    assert result.rule("together").violations == ()


def test_together_reports_who_works_without_the_other() -> None:
    dm = one_rule(pair_rule("together"))

    one = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r3"]}).rule("together")
    several = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "mon-pm": ["r2"], "mon-n": ["r1"]})

    assert outcome(one) == [("r1", ("mon-am",), None, 1)]
    assert one.violations[0].message == "r1 works mon-am without r2"
    assert outcome(several.rule("together")) == [
        ("r1", ("mon-n",), None, 1),
        ("r2", ("mon-pm",), None, 1),
    ]


def test_apart_reports_tasks_worked_by_both() -> None:
    dm = one_rule(pair_rule("apart", "hard"))

    zero = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r3"], "tue-am": ["r2", "r3"]})
    several = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "tue-am": ["r2", "r1"]})

    assert zero.rule("apart").violations == ()
    assert outcome(several.rule("apart")) == [
        (None, ("mon-am",), None, 1),
        (None, ("tue-am",), None, 1),
    ]
    assert several.rule("apart").violations[0].message == "r1 and r2 both work mon-am"
    assert several.score.hard == 2


def test_apart_and_together_respect_task_scope() -> None:
    mornings = {"tasks": [{"attribute": "type", "op": "eq", "value": "am"}]}
    dm = one_rule(pair_rule("apart", scope=mornings), pair_rule("together", scope=mornings))

    result = run(dm, RESOURCES, TASKS, {"tue-am": ["r1", "r2"], "mon-pm": ["r1"]})

    assert outcome(result.rule("apart")) == [(None, ("tue-am",), None, 1)]
    assert result.rule("together").violations == ()


def test_pair_rules_do_not_apply_when_a_named_resource_is_out_of_scope() -> None:
    full_time = {"resources": [{"attribute": "contract", "op": "eq", "value": "full_time"}]}
    dm = one_rule(pair_rule("apart", scope=full_time), pair_rule("together", scope=full_time))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "mon-pm": ["r1"]})

    assert result.rule("apart").violations == ()
    assert result.rule("together").violations == ()


# --- stability -------------------------------------------------------------------------------


def stability(*baseline: tuple[str, str], **fields: Any) -> dict[str, Any]:
    params = {"baseline": [{"task_id": t, "resource_id": r} for t, r in baseline]}
    return rule("stability", "soft", 5, params=params, **fields)


def test_stability_is_zero_when_the_baseline_is_kept() -> None:
    dm = one_rule(stability(("mon-am", "r1"), ("mon-n", "r3")))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r1", "r2"], "mon-n": ["r3"]})

    assert result.rule("stability").violations == ()


def test_stability_reports_each_missing_baseline_pair_once() -> None:
    dm = one_rule(stability(("mon-am", "r1"), ("mon-n", "r3"), ("mon-n", "r3"), ("tue-am", "r2")))

    result = run(dm, RESOURCES, TASKS, {"mon-am": ["r2"], "tue-am": ["r2"]})

    assert outcome(result.rule("stability")) == [
        ("r1", ("mon-am",), None, 1),
        ("r3", ("mon-n",), None, 1),
    ]
    assert result.rule("stability").violations[0].message == (
        "r1 no longer works mon-am (baseline)"
    )
    assert result.score.soft == 10


def test_stability_ignores_out_of_scope_baseline_pairs() -> None:
    scope = {"resources": [{"attribute": "id", "op": "ne", "value": "r3"}]}
    dm = one_rule(stability(("mon-am", "r1"), ("mon-n", "r3"), scope=scope))

    result = run(dm, RESOURCES, TASKS, {})

    assert outcome(result.rule("stability")) == [("r1", ("mon-am",), None, 1)]


# --- scope -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tasks_scope", "expected"),
    [
        ([{"attribute": "weekday", "op": "eq", "value": "tue"}], ["tue-am", "tue-n"]),
        ([{"attribute": "start_hour", "op": "gte", "value": 15}], ["mon-n", "mon-pm", "tue-n"]),
        (
            [{"attribute": "date", "op": "lte", "value": "2026-11-02"}],
            ["mon-am", "mon-n", "mon-pm"],
        ),
        ([{"attribute": "duration_hours", "op": "eq", "value": 8}], sorted(t.id for t in TASKS)),
        ([{"attribute": "required_skill", "op": "ne", "value": "x"}], ["mon-n", "tue-n"]),
    ],
)
def test_derived_and_data_attributes_select_tasks(
    tasks_scope: list[dict[str, Any]], expected: list[str]
) -> None:
    rule_doc = rule("preference", "soft", 1, params={"kind": "avoid"}, scope={"tasks": tasks_scope})
    dm = one_rule(rule_doc)

    result = run(dm, RESOURCES, TASKS, {t.id: ["r1"] for t in TASKS})

    assert sorted(v.task_ids[0] for v in result.rule("preference").violations) == expected


def test_rule_restricted_by_resource_and_task_filters_ignores_everything_else() -> None:
    scope = {
        "resources": [{"attribute": "contract", "op": "eq", "value": "full_time"}],
        "tasks": [
            {"attribute": "weekday", "op": "in", "value": ["mon"]},
            {"attribute": "type", "op": "not_in", "value": ["night"]},
        ],
    }
    dm = one_rule(limit_rule("max_per_period", "count", "horizon", 0, scope=scope))
    everyone_everywhere = {t.id: ["r1", "r2", "r3"] for t in TASKS}

    result = run(dm, RESOURCES, TASKS, everyone_everywhere)

    assert outcome(result.rule("max_per_period")) == [
        ("r1", ("mon-am", "mon-pm"), "2026-11-02/2026-11-08", 2),
        ("r3", ("mon-am", "mon-pm"), "2026-11-02/2026-11-08", 2),
    ]
