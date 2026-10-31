"""The readable text report of a check result."""

from checker_helpers import make_dm, res, rule, run, task

from potakt.checker import render_check_result
from potakt.model import DecisionModel

RESOURCES = [res("r1"), res("r2")]
TASKS = [
    task("t1", "2026-11-02T07:00", "2026-11-02T15:00"),
    task("t2", "2026-11-02T14:00", "2026-11-02T22:00"),
]


def dm_with_title(title: str) -> DecisionModel:
    dm = make_dm(
        rule("no_overlap"),
        rule("preference", "soft", 3, params={"kind": "avoid"}, id="avoid-all"),
        rule("apart", "soft", 2, params={"resource_ids": ["r1", "r2"]}),
    )
    return dm.model_copy(update={"title": title})


def test_report_shows_score_then_violated_rules_then_clean_rules() -> None:
    dm = dm_with_title("Βάρδιες")

    report = render_check_result(run(dm, RESOURCES, TASKS, {"t1": ["r1"], "t2": ["r1"]}), dm)

    assert report == (
        "Βάρδιες (dm-test v1)\n"
        "Score: hard 1, soft 6 (infeasible)\n"
        "\n"
        "no_overlap — no_overlap, hard: amount 1, hard +1\n"
        "  - r1: t1 overlaps t2\n"
        "\n"
        "avoid-all — preference, soft, weight 3: amount 2, soft +6\n"
        "  - r1 on t1: avoided assignment\n"
        "  - r1 on t2: avoided assignment\n"
        "\n"
        "Rules without violations: coverage, apart\n"
    )


def test_report_of_a_clean_solution_lists_rules_compactly() -> None:
    dm = make_dm(rule("no_overlap"))

    report = render_check_result(run(dm, RESOURCES, TASKS, {"t1": ["r1"], "t2": ["r2"]}), dm)

    assert report == (
        "Test model (dm-test v1)\n"
        "Score: hard 0, soft 0 (feasible)\n"
        "\n"
        "Rules without violations: coverage, no_overlap\n"
    )


def test_report_shows_rewards_with_a_minus_sign() -> None:
    dm = make_dm(rule("preference", "soft", 2, params={"kind": "prefer"}, id="likes"))

    report = render_check_result(run(dm, RESOURCES, TASKS, {"t1": ["r1"], "t2": ["r2"]}), dm)

    assert "Score: hard 0, soft -4 (feasible)\n" in report
    assert "likes — preference, soft, weight 2: amount 2, soft -4\n" in report
