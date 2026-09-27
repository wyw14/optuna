from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import pytest

import optuna
from optuna.study import AlignedIntermediateValue
from optuna.study import AlignedIntermediateValues
from optuna.study import AlignedTrialIntermediateValues
from optuna.study import AlignedValueType
from optuna.study._intermediate_alignment import align_intermediate_values
from optuna.testing.storages import StorageSupplier
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _add_curve(
    study: optuna.Study,
    curve: dict[int, float],
    *,
    state: TrialState = TrialState.COMPLETE,
) -> None:
    value = 0.0 if state == TrialState.COMPLETE else None
    study.add_trial(create_trial(state=state, value=value, intermediate_values=dict(curve)))


def _make_study(curves: Sequence[dict[int, float]]) -> optuna.Study:
    study = optuna.create_study()
    for curve in curves:
        _add_curve(study, curve)
    return study


def _kinds(trial: AlignedTrialIntermediateValues) -> list[str]:
    return [point.kind.name for point in trial.points]


def test_module_level_function_matches_method() -> None:
    study = _make_study([{0: 0.0, 2: 2.0}, {0: 0.0, 1: 1.0, 2: 2.0}])

    by_method = study.align_intermediate_values(target_steps=[0, 1, 2])
    by_function = align_intermediate_values(study, target_steps=[0, 1, 2])
    assert by_method == by_function
    assert isinstance(by_method, AlignedIntermediateValues)


def test_default_target_steps_are_union_of_reports() -> None:
    study = _make_study([{0: 0.0, 2: 2.0}, {1: 1.0, 3: 3.0}])

    aligned = study.align_intermediate_values()

    assert aligned.target_steps == (0, 1, 2, 3)
    assert aligned.interpolation == "linear"
    assert [trial.trial_number for trial in aligned.trials] == [0, 1]


def test_reported_points_keep_exact_values() -> None:
    study = _make_study([{0: 0.0, 2: 4.0}])

    aligned = study.align_intermediate_values(target_steps=[0, 2])

    (trial,) = aligned.trials
    assert _kinds(trial) == ["REPORTED", "REPORTED"]
    assert trial.values == (0.0, 4.0)
    assert all(point.interpolation is None for point in trial.points)


def test_linear_interpolation_between_reports() -> None:
    study = _make_study([{0: 0.0, 4: 8.0}])

    aligned = study.align_intermediate_values(target_steps=[0, 1, 2, 3, 4])

    (trial,) = aligned.trials
    assert trial.values == (0.0, 2.0, 4.0, 6.0, 8.0)
    assert _kinds(trial) == [
        "REPORTED",
        "INTERPOLATED",
        "INTERPOLATED",
        "INTERPOLATED",
        "REPORTED",
    ]
    assert trial.points[1].interpolation == "linear"
    assert trial.points[2].value == 4.0


def test_no_extrapolation_beyond_support() -> None:
    # The trial reports at steps 1 and 3: nothing exists outside [1, 3].
    study = _make_study([{1: 2.0, 3: 6.0}])

    aligned = study.align_intermediate_values(target_steps=[0, 1, 2, 3, 4])

    (trial,) = aligned.trials
    assert trial.first_step == 1
    assert trial.last_step == 3
    assert trial.values == (None, 2.0, 4.0, 6.0, None)
    assert _kinds(trial) == [
        "UNAVAILABLE",
        "REPORTED",
        "INTERPOLATED",
        "REPORTED",
        "UNAVAILABLE",
    ]
    for step, point in zip((0, 1, 2, 3, 4), trial.points):
        assert point.step == step
    # Unavailable points carry no value and no interpolation rule.
    assert trial.points[0].value is None
    assert trial.points[0].interpolation is None
    assert trial.points[4].value is None
    assert trial.points[4].interpolation is None


def test_missing_intervals_are_never_zero() -> None:
    study = _make_study([{0: 10.0}])

    aligned = study.align_intermediate_values(target_steps=[0, 1, 2])

    (trial,) = aligned.trials
    assert trial.values == (10.0, None, None)
    assert all(value != 0.0 for value in trial.values if value is not None)


def test_single_report_trial() -> None:
    study = _make_study([{2: 5.0}])

    aligned = study.align_intermediate_values(target_steps=[0, 1, 2, 3])

    (trial,) = aligned.trials
    assert trial.values == (None, None, 5.0, None)
    assert _kinds(trial) == [
        "UNAVAILABLE",
        "UNAVAILABLE",
        "REPORTED",
        "UNAVAILABLE",
    ]


def test_budget_range_restricts_target_points() -> None:
    study = _make_study([{0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}])

    assert study.align_intermediate_values(budget_range=(1, 3)).target_steps == (1, 2, 3)
    assert study.align_intermediate_values(budget_range=(None, 2)).target_steps == (0, 1, 2)
    assert study.align_intermediate_values(budget_range=(3, None)).target_steps == (3, 4)

    # The range only clips target points; it never triggers extrapolation.
    aligned = study.align_intermediate_values(target_steps=[0, 2, 9], budget_range=(0, 4))
    assert aligned.target_steps == (0, 2)


def test_pruned_trial_keeps_reported_part() -> None:
    study = optuna.create_study()
    _add_curve(study, {0: 0.0, 2: 4.0, 4: 8.0}, state=TrialState.COMPLETE)
    _add_curve(study, {0: 0.0, 2: 2.0}, state=TrialState.PRUNED)

    aligned = study.align_intermediate_values(target_steps=[0, 1, 2, 3, 4])

    pruned = aligned.trials[1]
    assert pruned.state == TrialState.PRUNED
    assert pruned.last_step == 2
    assert pruned.values == (0.0, 1.0, 2.0, None, None)
    assert _kinds(pruned) == [
        "REPORTED",
        "INTERPOLATED",
        "REPORTED",
        "UNAVAILABLE",
        "UNAVAILABLE",
    ]


def test_failed_trial_keeps_reported_part() -> None:
    study = optuna.create_study()
    _add_curve(study, {0: 1.0, 2: 3.0}, state=TrialState.FAIL)

    aligned = study.align_intermediate_values(target_steps=[0, 1, 2])

    (failed,) = aligned.trials
    assert failed.state == TrialState.FAIL
    assert failed.values == (1.0, 2.0, 3.0)
    assert _kinds(failed) == ["REPORTED", "INTERPOLATED", "REPORTED"]


def test_running_trial_is_included() -> None:
    study = optuna.create_study()
    _add_curve(study, {0: 0.0, 1: 1.0}, state=TrialState.RUNNING)

    aligned = study.align_intermediate_values(target_steps=[0, 1])

    assert len(aligned.trials) == 1
    assert aligned.trials[0].state == TrialState.RUNNING


def test_trials_without_reports_are_excluded() -> None:
    study = optuna.create_study()
    _add_curve(study, {0: 0.0})
    study.add_trial(create_trial(value=1.0))

    aligned = study.align_intermediate_values(target_steps=[0])

    assert [trial.trial_number for trial in aligned.trials] == [0]


def test_states_filter() -> None:
    study = optuna.create_study()
    _add_curve(study, {0: 0.0}, state=TrialState.COMPLETE)
    _add_curve(study, {0: 0.0}, state=TrialState.FAIL)

    complete_only = study.align_intermediate_values(
        target_steps=[0], states=(TrialState.COMPLETE,)
    )
    assert [trial.state for trial in complete_only.trials] == [TrialState.COMPLETE]

    all_states = study.align_intermediate_values(target_steps=[0], states=None)
    assert {trial.state for trial in all_states.trials} == {
        TrialState.COMPLETE,
        TrialState.FAIL,
    }


def test_trial_numbers_filter() -> None:
    study = _make_study([{0: 0.0}, {0: 1.0}, {0: 2.0}])

    aligned = study.align_intermediate_values(target_steps=[0], trial_numbers=(0, 2))

    assert [trial.trial_number for trial in aligned.trials] == [0, 2]


def test_incompatible_budget_semantics_raise() -> None:
    study = _make_study([{0: 0.0}, {0: 0.0}, {0: 0.0}])

    with pytest.raises(ValueError, match="budget semantics"):
        study.align_intermediate_values(
            target_steps=[0], budget_semantics={0: "epoch", 1: "epoch", 2: "iteration"}
        )

    # Trials absent from the mapping fall back to the default unitless "step" semantics and
    # therefore also conflict with explicitly labeled trials.
    with pytest.raises(ValueError, match="separately"):
        study.align_intermediate_values(target_steps=[0], budget_semantics={2: "iteration"})


def test_compatible_budget_semantics() -> None:
    study = _make_study([{0: 0.0}, {0: 0.0}])

    aligned = study.align_intermediate_values(
        target_steps=[0], budget_semantics={0: "epoch", 1: "epoch"}
    )
    assert aligned.budget_semantics == "epoch"

    defaulted = study.align_intermediate_values(target_steps=[0])
    assert defaulted.budget_semantics is None

    defaulted_with_mapping = study.align_intermediate_values(
        target_steps=[0], budget_semantics={0: "step"}
    )
    assert defaulted_with_mapping.budget_semantics == "step"


def test_separate_comparison_with_trial_numbers() -> None:
    study = _make_study([{0: 0.0}, {0: 0.0}])
    semantics = {0: "epoch", 1: "iteration"}

    # Groups with a single, homogeneous semantics can be compared separately.
    epochs = study.align_intermediate_values(
        target_steps=[0], trial_numbers=(0,), budget_semantics=semantics
    )
    iterations = study.align_intermediate_values(
        target_steps=[0], trial_numbers=(1,), budget_semantics=semantics
    )
    assert [t.trial_number for t in epochs.trials] == [0]
    assert [t.trial_number for t in iterations.trials] == [1]
    assert epochs.budget_semantics == "epoch"
    assert iterations.budget_semantics == "iteration"


def test_values_and_kinds_properties() -> None:
    study = _make_study([{0: 0.0, 2: 2.0}])

    aligned = study.align_intermediate_values(target_steps=[0, 1, 2])
    (trial,) = aligned.trials

    assert trial.values == (0.0, 1.0, 2.0)
    assert trial.kinds == (
        AlignedValueType.REPORTED,
        AlignedValueType.INTERPOLATED,
        AlignedValueType.REPORTED,
    )


def test_empty_study() -> None:
    study = optuna.create_study()

    aligned = study.align_intermediate_values()

    assert isinstance(aligned, AlignedIntermediateValues)
    assert aligned.target_steps == ()
    assert aligned.trials == ()

    # Providing semantics for a study without any (reporting) trial must not raise either.
    also_aligned = study.align_intermediate_values(budget_semantics={})
    assert also_aligned.trials == ()
    assert also_aligned.budget_semantics is None


@pytest.mark.parametrize(
    "target_steps",
    [
        [0, 0],
        [2, 1],
        [1.0, 2],  # type: ignore[list-item]
        [True],  # type: ignore[list-item]
    ],
)
def test_invalid_target_steps(target_steps: Sequence[int]) -> None:
    study = _make_study([{0: 0.0}])

    with pytest.raises(ValueError):
        study.align_intermediate_values(target_steps=target_steps)


@pytest.mark.parametrize(
    "budget_range",
    [
        (2, 1),
        (1,),
        ("0", 1),
    ],
)
def test_invalid_budget_range(budget_range: tuple[object, object]) -> None:
    study = _make_study([{0: 0.0}])

    with pytest.raises(ValueError):
        study.align_intermediate_values(budget_range=budget_range)  # type: ignore[arg-type]


def test_unsupported_interpolation_rule() -> None:
    study = _make_study([{0: 0.0}])

    with pytest.raises(ValueError, match="linear"):
        study.align_intermediate_values(target_steps=[0], interpolation="nearest")


def test_output_objects_are_read_only() -> None:
    study = _make_study([{0: 0.0, 2: 2.0}])
    aligned = study.align_intermediate_values(target_steps=[0, 1, 2])

    with pytest.raises(dataclasses.FrozenInstanceError):
        aligned.target_steps = (0,)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        aligned.trials[0].points[0].value = 1.0  # type: ignore[misc]
    assert isinstance(aligned.trials[0], AlignedTrialIntermediateValues)
    assert isinstance(aligned.trials[0].points[0], AlignedIntermediateValue)


@pytest.mark.parametrize("storage_mode", ["inmemory", "sqlite"])
def test_alignment_does_not_write_back_to_storage(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        _add_curve(study, {0: 0.0, 2: 4.0}, state=TrialState.COMPLETE)
        _add_curve(study, {0: 0.0, 2: 2.0}, state=TrialState.PRUNED)

        written_calls: list[tuple[object, ...]] = []
        original_set = storage.set_trial_intermediate_value

        def spy(*args: object, **kwargs: object) -> None:
            written_calls.append(args)
            original_set(*args, **kwargs)

        storage.set_trial_intermediate_value = spy  # type: ignore[method-assign]

        study.align_intermediate_values(target_steps=[0, 1, 2, 3])

        assert written_calls == []

        reloaded_trials = storage.get_all_trials(study._study_id, deepcopy=True)
        assert {t.number: dict(t.intermediate_values) for t in reloaded_trials} == {
            0: {0: 0.0, 2: 4.0},
            1: {0: 0.0, 2: 2.0},
        }
        for trial in reloaded_trials:
            assert all(isinstance(v, float) for v in trial.intermediate_values.values())


def test_alignment_does_not_affect_pruning() -> None:
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -1.0, 1.0)
        for step in range(4):
            trial.report(x * (4 - step), step)
            if trial.should_prune():
                raise optuna.TrialPruned
        return x * 4.0

    # Aligning after every trial must not change reported values or pruning decisions.
    snapshots: list[dict[int, dict[int, float]]] = []

    def callback(study_: optuna.Study, _trial: FrozenTrial) -> None:
        before = {t.number: dict(t.intermediate_values) for t in study_.get_trials(deepcopy=False)}
        study_.align_intermediate_values()
        after = {t.number: dict(t.intermediate_values) for t in study_.get_trials(deepcopy=False)}
        assert before == after
        snapshots.append(after)

    study.optimize(objective, n_trials=6, callbacks=[callback])

    assert len(snapshots) == 6
    # Only the steps actually reported inside the objective remain in storage; no aligned
    # or interpolated point ever leaks back into the trials.
    for trial in study.get_trials():
        assert set(trial.intermediate_values) <= set(range(4))
