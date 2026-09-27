from __future__ import annotations

import datetime

import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import ExternalTrial
from optuna.study import PrecheckIssueCategory
from optuna.study import StudyDirection
from optuna.study import TrialImportStatus
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


# Inmemory and journal-file are always available; sqlite additionally needs sqlalchemy.
_STORAGE_MODES = ["inmemory", "journal"]
try:
    import sqlalchemy  # noqa: F401

    _STORAGE_MODES.append("sqlite")
except ImportError:
    pass


def _make_storage(mode: str, tmp_path):
    if mode == "inmemory":
        return optuna.storages.InMemoryStorage()
    if mode == "journal":
        log_path = str(tmp_path / "journal.log")
        # Use the open(2)-based lock because the symlink lock requires special privileges on
        # Windows.
        lock = optuna.storages.journal.JournalFileOpenLock(log_path)
        return optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(log_path, lock_obj=lock)
        )
    if mode == "sqlite":
        return optuna.storages.RDBStorage(f"sqlite:///{tmp_path / 'test.db'}")
    raise AssertionError(f"Unknown storage mode: {mode}")


_UNSET = object()


def _complete_trial(
    value: float | None = None,
    *,
    values: list[float] | None = None,
    params: dict | None = None,
    distributions: dict | None = None,
    state: TrialState = TrialState.COMPLETE,
    datetime_start=_UNSET,
    datetime_complete=_UNSET,
) -> FrozenTrial:
    if state == TrialState.COMPLETE and value is not None and values is None:
        values = [value]
    if datetime_start is _UNSET:
        datetime_start = None if state == TrialState.WAITING else datetime.datetime(2024, 1, 1)
    if datetime_complete is _UNSET:
        datetime_complete = datetime_start if state.is_finished() else None
    return FrozenTrial(
        number=-1,
        state=state,
        value=None,
        values=values,
        datetime_start=datetime_start,
        datetime_complete=datetime_complete,
        params=params if params is not None else {},
        distributions=distributions if distributions is not None else {},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=-1,
    )


def _codes(result) -> list[tuple[str, PrecheckIssueCategory]]:
    return [(issue.code, issue.category) for issue in result.issues]


def _has_issue(result, code: str, category: PrecheckIssueCategory) -> bool:
    return (code, category) in _codes(result)


@pytest.mark.parametrize("storage_mode", _STORAGE_MODES)
def test_precheck_does_not_write_storage(storage_mode: str, tmp_path) -> None:
    storage = _make_storage(storage_mode, tmp_path)
    study = optuna.create_study(storage=storage, direction="minimize")
    study.add_trial(
        create_trial(
            params={"x": 1.0},
            distributions={"x": FloatDistribution(0, 10)},
            value=1.0,
        )
    )
    n_trials_before = len(study.trials)

    trial = create_trial(
        params={"x": 2.0},
        distributions={"x": FloatDistribution(0, 10)},
        value=4.0,
    )
    report = study.precheck_add_trials(
        [ExternalTrial(trial, directions=[StudyDirection.MINIMIZE])]
    )

    # Nothing was written.
    assert len(study.trials) == n_trials_before
    assert len(study._storage.get_all_trials(study._study_id)) == n_trials_before

    # The report is bound to the observed target study.
    assert report.study_name == study.study_name
    assert report.directions == (StudyDirection.MINIMIZE,)
    assert report.n_trials == 1
    assert set(report.distributions.keys()) == {"x"}
    assert report.is_premise_current(study)

    result = report.results[0]
    assert result.index == 0
    assert result.status == TrialImportStatus.IMPORTABLE
    assert result.issues == ()
    assert report.all_importable
    assert report.importable_indices == (0,)
    assert report.blocked_indices == ()
    assert report.unconfirmed_indices == ()


def test_importable_record_with_directions() -> None:
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        create_trial(
            params={"x": 1.0},
            distributions={"x": FloatDistribution(0, 10)},
            value=1.0,
        )
    )
    trial = create_trial(
        params={"x": 2.0},
        distributions={"x": FloatDistribution(0, 10)},
        value=4.0,
    )
    report = study.precheck_add_trials(
        [ExternalTrial(trial, directions=[StudyDirection.MINIMIZE])]
    )
    assert report.results[0].status == TrialImportStatus.IMPORTABLE


def test_bare_trial_direction_unconfirmed() -> None:
    study = optuna.create_study(direction="minimize")
    valued_trial = create_trial(
        params={"x": 1.0}, distributions={"x": FloatDistribution(0, 1)}, value=1.0
    )
    failed_trial = _complete_trial(values=None, state=TrialState.FAIL)
    report = study.precheck_add_trials([valued_trial, failed_trial])

    assert report.results[0].status == TrialImportStatus.UNCONFIRMED
    assert _has_issue(
        report.results[0], "direction_unconfirmed", PrecheckIssueCategory.UNVERIFIED
    )
    assert report.unconfirmed_indices == (0,)
    # A record without objective values does not need any direction provenance.
    assert report.results[1].status == TrialImportStatus.IMPORTABLE
    assert report.all_importable is False


def test_objective_count_mismatch() -> None:
    study = optuna.create_study(direction="minimize")
    multi_trial = create_trial(
        values=[1.0, 2.0],
        params={"x": 1.0},
        distributions={"x": FloatDistribution(0, 1)},
    )
    report = study.precheck_add_trials(
        [ExternalTrial(multi_trial, directions=[StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])]
    )
    result = report.results[0]
    assert result.status == TrialImportStatus.CONFLICT
    assert _has_issue(
        result, "objective_count_mismatch", PrecheckIssueCategory.TARGET_STUDY
    )
    assert report.blocked_indices == (0,)

    mo_study = optuna.create_study(directions=["minimize", "maximize"])
    single_trial = create_trial(
        value=1.0, params={"x": 1.0}, distributions={"x": FloatDistribution(0, 1)}
    )
    report = mo_study.precheck_add_trials(
        [ExternalTrial(single_trial, directions=[StudyDirection.MINIMIZE])]
    )
    assert _has_issue(
        report.results[0],
        "objective_count_mismatch",
        PrecheckIssueCategory.TARGET_STUDY,
    )


def test_direction_mismatch() -> None:
    study = optuna.create_study(direction="minimize")
    trial = create_trial(
        params={"x": 1.0}, distributions={"x": FloatDistribution(0, 1)}, value=1.0
    )
    report = study.precheck_add_trials(
        [ExternalTrial(trial, directions=[StudyDirection.MAXIMIZE])]
    )
    result = report.results[0]
    assert result.status == TrialImportStatus.CONFLICT
    mismatch = [
        issue
        for issue in result.issues
        if issue.code == "direction_mismatch"
        and issue.category == PrecheckIssueCategory.TARGET_STUDY
    ]
    assert len(mismatch) == 1
    assert mismatch[0].objective == 0

    mo_study = optuna.create_study(directions=["minimize", "maximize"])
    mo_trial = create_trial(
        values=[1.0, 2.0],
        params={"x": 1.0},
        distributions={"x": FloatDistribution(0, 1)},
    )
    report = mo_study.precheck_add_trials(
        [
            ExternalTrial(
                mo_trial, directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]
            )
        ]
    )
    result = report.results[0]
    assert result.status == TrialImportStatus.CONFLICT
    objectives = [issue.objective for issue in result.issues if issue.code == "direction_mismatch"]
    assert objectives == [1]


@pytest.mark.parametrize("storage_mode", _STORAGE_MODES)
def test_dynamic_numerical_space_is_accepted(storage_mode: str, tmp_path) -> None:
    storage = _make_storage(storage_mode, tmp_path)
    study = optuna.create_study(storage=storage, direction="minimize")
    study.add_trial(
        create_trial(
            params={"x": 5.0, "y": 3},
            distributions={"x": FloatDistribution(0, 10), "y": IntDistribution(0, 10)},
            value=1.0,
        )
    )

    widened = create_trial(
        params={"x": -2.0, "y": 12},
        distributions={"x": FloatDistribution(-5, 20), "y": IntDistribution(0, 20)},
        value=1.0,
    )
    narrowed = create_trial(
        params={"x": 1.0, "y": 2},
        distributions={"x": FloatDistribution(0, 5), "y": IntDistribution(0, 5)},
        value=1.0,
    )
    with_step = create_trial(
        params={"x": 1.5},
        distributions={"x": FloatDistribution(0, 10, step=0.5)},
        value=1.0,
    )
    report = study.precheck_add_trials(
        [
            ExternalTrial(t, directions=[StudyDirection.MINIMIZE])
            for t in (widened, narrowed, with_step)
        ]
    )
    assert report.all_importable
    # The observed target space is not mutated by the precheck.
    assert report.distributions["x"] == FloatDistribution(0, 10)


def test_distribution_kind_and_log_conflict_with_target() -> None:
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        create_trial(
            params={"x": 5.0}, distributions={"x": FloatDistribution(0, 10)}, value=1.0
        )
    )

    kind_trial = create_trial(
        params={"x": 3}, distributions={"x": IntDistribution(0, 10)}, value=1.0
    )
    log_trial = create_trial(
        params={"x": 5.0}, distributions={"x": FloatDistribution(1, 10, log=True)}, value=1.0
    )
    report = study.precheck_add_trials(
        [
            ExternalTrial(kind_trial, directions=[StudyDirection.MINIMIZE]),
            ExternalTrial(log_trial, directions=[StudyDirection.MINIMIZE]),
        ]
    )
    for result in report.results:
        assert result.status == TrialImportStatus.CONFLICT
        assert _has_issue(
            result, "distribution_conflict", PrecheckIssueCategory.TARGET_STUDY
        )
        assert result.issues[0].param_name == "x"


def test_categorical_choices_conflict_with_target() -> None:
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        create_trial(
            params={"c": "a"},
            distributions={"c": CategoricalDistribution(("a", "b"))},
            value=1.0,
        )
    )
    reordered = create_trial(
        params={"c": "a"},
        distributions={"c": CategoricalDistribution(("b", "a"))},
        value=1.0,
    )
    extended = create_trial(
        params={"c": "a"},
        distributions={"c": CategoricalDistribution(("a", "b", "c"))},
        value=1.0,
    )
    report = study.precheck_add_trials(
        [
            ExternalTrial(reordered, directions=[StudyDirection.MINIMIZE]),
            ExternalTrial(extended, directions=[StudyDirection.MINIMIZE]),
        ]
    )
    for result in report.results:
        assert result.status == TrialImportStatus.CONFLICT
        assert _has_issue(
            result, "distribution_conflict", PrecheckIssueCategory.TARGET_STUDY
        )


def test_new_parameter_is_importable() -> None:
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        create_trial(
            params={"x": 1.0}, distributions={"x": FloatDistribution(0, 1)}, value=1.0
        )
    )
    new_param_trial = create_trial(
        params={"x": 1.0, "z": "a"},
        distributions={
            "x": FloatDistribution(0, 1),
            "z": CategoricalDistribution(("a", "b")),
        },
        value=1.0,
    )
    report = study.precheck_add_trials(
        [ExternalTrial(new_param_trial, directions=[StudyDirection.MINIMIZE])]
    )
    assert report.results[0].status == TrialImportStatus.IMPORTABLE
    assert "z" not in report.distributions


@pytest.mark.parametrize(
    "trial,expected_code",
    [
        (
            _complete_trial(values=None),
            "complete_trial_missing_values",
        ),
        (
            _complete_trial(values=[1.0], state=TrialState.FAIL),
            "failed_trial_has_values",
        ),
        (
            _complete_trial(values=[float("nan")]),
            "nan_objective_value",
        ),
        (
            _complete_trial(
                values=None,
                state=TrialState.RUNNING,
                datetime_start=None,
                datetime_complete=None,
            ),
            "missing_datetime_start",
        ),
        (
            _complete_trial(values=None, state=TrialState.PRUNED, datetime_complete=None),
            "missing_datetime_complete",
        ),
        (
            _complete_trial(
                values=None,
                state=TrialState.RUNNING,
                datetime_complete=datetime.datetime(2024, 1, 1),
            ),
            "unexpected_datetime_complete",
        ),
        (
            _complete_trial(
                values=[1.0],
                params={"x": 1.0},
                distributions={"y": FloatDistribution(0, 1)},
            ),
            "params_distributions_key_mismatch",
        ),
        (
            _complete_trial(
                values=[1.0],
                params={"x": 5.0},
                distributions={"x": FloatDistribution(0, 1)},
            ),
            "param_value_out_of_distribution",
        ),
        (
            _complete_trial(
                values=[1.0],
                params={"x": 1.0},
                distributions={"x": "not-a-distribution"},
            ),
            "invalid_distribution",
        ),
    ],
)
def test_self_invalid_records(trial: FrozenTrial, expected_code: str) -> None:
    study = optuna.create_study(direction="minimize")
    report = study.precheck_add_trials(
        [ExternalTrial(trial, directions=[StudyDirection.MINIMIZE])]
    )
    result = report.results[0]
    assert result.status == TrialImportStatus.CONFLICT
    assert _has_issue(result, expected_code, PrecheckIssueCategory.SELF)
    # Such records are rejected by add_trial as well.
    with pytest.raises((ValueError, TypeError, AttributeError)):
        study.add_trial(trial)


def test_batch_distribution_kind_conflict() -> None:
    study = optuna.create_study(direction="minimize")
    first = create_trial(
        params={"x": 0.5}, distributions={"x": FloatDistribution(0, 1)}, value=1.0
    )
    second = create_trial(
        params={"x": "a"},
        distributions={"x": CategoricalDistribution(("a", "b"))},
        value=1.0,
    )
    report = study.precheck_add_trials(
        [
            ExternalTrial(first, directions=[StudyDirection.MINIMIZE]),
            ExternalTrial(second, directions=[StudyDirection.MINIMIZE]),
        ]
    )
    assert report.results[0].status == TrialImportStatus.IMPORTABLE
    result = report.results[1]
    assert result.status == TrialImportStatus.CONFLICT
    issue = next(
        issue
        for issue in result.issues
        if issue.code == "batch_distribution_conflict"
    )
    assert issue.category == PrecheckIssueCategory.BATCH
    assert issue.conflicting_index == 0
    assert issue.param_name == "x"


def test_batch_categorical_choices_conflict_and_order() -> None:
    study = optuna.create_study(direction="minimize")
    first = create_trial(
        params={"c": "a"},
        distributions={"c": CategoricalDistribution(("a", "b"))},
        value=1.0,
    )
    second = create_trial(
        params={"c": "a"},
        distributions={"c": CategoricalDistribution(("a", "c"))},
        value=1.0,
    )
    third = create_trial(
        params={"c": "a"},
        distributions={"c": CategoricalDistribution(("a", "b"))},
        value=1.0,
    )
    report = study.precheck_add_trials(
        [
            ExternalTrial(t, directions=[StudyDirection.MINIMIZE])
            for t in (first, second, third)
        ]
    )
    assert report.results[0].status == TrialImportStatus.IMPORTABLE
    assert report.results[1].status == TrialImportStatus.CONFLICT
    assert _has_issue(
        report.results[1],
        "batch_distribution_conflict",
        PrecheckIssueCategory.BATCH,
    )
    # The conflicting second record does not establish a new space; the third record matches
    # the first and is importable.
    assert report.results[2].status == TrialImportStatus.IMPORTABLE
    assert report.results[1].issues[0].conflicting_index == 0

    # Reversing the order flips which record is reported.
    report = study.precheck_add_trials(
        [
            ExternalTrial(second, directions=[StudyDirection.MINIMIZE]),
            ExternalTrial(first, directions=[StudyDirection.MINIMIZE]),
        ]
    )
    assert report.results[0].status == TrialImportStatus.IMPORTABLE
    assert report.results[1].status == TrialImportStatus.CONFLICT
    assert report.results[1].issues[0].conflicting_index == 0


def test_batch_dynamic_bounds_accepted_but_log_conflicts() -> None:
    study = optuna.create_study(direction="minimize")
    narrow = create_trial(
        params={"x": 0.5}, distributions={"x": FloatDistribution(0, 1)}, value=1.0
    )
    wide = create_trial(
        params={"x": 5.0}, distributions={"x": FloatDistribution(0, 10)}, value=1.0
    )
    logged = create_trial(
        params={"x": 5.0},
        distributions={"x": FloatDistribution(1, 10, log=True)},
        value=1.0,
    )
    report = study.precheck_add_trials(
        [
            ExternalTrial(t, directions=[StudyDirection.MINIMIZE])
            for t in (narrow, wide, logged)
        ]
    )
    assert report.results[0].status == TrialImportStatus.IMPORTABLE
    assert report.results[1].status == TrialImportStatus.IMPORTABLE
    assert report.results[2].status == TrialImportStatus.CONFLICT
    # The introducer of the parameter is record 0, even though record 1 widened the bounds.
    assert report.results[2].issues[0].conflicting_index == 0


def test_batch_conflict_origin_skips_self_invalid_record() -> None:
    study = optuna.create_study(direction="minimize")
    invalid = _complete_trial(
        values=[1.0],
        params={"x": 5.0},
        distributions={"x": FloatDistribution(0, 1)},
    )
    categorical = create_trial(
        params={"x": "a"},
        distributions={"x": CategoricalDistribution(("a", "b"))},
        value=1.0,
    )
    report = study.precheck_add_trials(
        [
            ExternalTrial(invalid, directions=[StudyDirection.MINIMIZE]),
            ExternalTrial(categorical, directions=[StudyDirection.MINIMIZE]),
        ]
    )
    # The self-invalid record does not establish the space, so the second record cannot clash
    # with it and is only unconfirmed-independent (it provides directions, hence importable).
    assert report.results[0].status == TrialImportStatus.CONFLICT
    assert _has_issue(
        report.results[0],
        "param_value_out_of_distribution",
        PrecheckIssueCategory.SELF,
    )
    assert report.results[1].status == TrialImportStatus.IMPORTABLE


def test_invalid_source_directions() -> None:
    study = optuna.create_study(direction="minimize")
    trial = create_trial(
        params={"x": 1.0}, distributions={"x": FloatDistribution(0, 1)}, value=1.0
    )

    # Length mismatch with the record's own values: self-inconsistent metadata.
    report = study.precheck_add_trials(
        [ExternalTrial(trial, directions=[StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])]
    )
    assert _has_issue(
        report.results[0],
        "invalid_source_directions",
        PrecheckIssueCategory.SELF,
    )
    assert report.results[0].status == TrialImportStatus.CONFLICT

    # NOT_SET is not a meaningful direction.
    report = study.precheck_add_trials(
        [ExternalTrial(trial, directions=[StudyDirection.NOT_SET])]
    )
    assert _has_issue(
        report.results[0],
        "invalid_source_directions",
        PrecheckIssueCategory.SELF,
    )

    # Non-StudyDirection values.
    report = study.precheck_add_trials([ExternalTrial(trial, directions=["minimize"])])
    assert _has_issue(
        report.results[0],
        "invalid_source_directions",
        PrecheckIssueCategory.SELF,
    )


def test_waiting_and_pruned_records() -> None:
    study = optuna.create_study(direction="minimize")

    waiting = _complete_trial(values=None, state=TrialState.WAITING, datetime_start=None)
    pruned_with_value = create_trial(
        state=TrialState.PRUNED,
        params={"x": 1.0},
        distributions={"x": FloatDistribution(0, 1)},
        value=0.5,
    )
    pruned_without_value = _complete_trial(values=None, state=TrialState.PRUNED)
    report = study.precheck_add_trials(
        [
            waiting,
            ExternalTrial(pruned_with_value, directions=[StudyDirection.MINIMIZE]),
            pruned_without_value,
        ]
    )
    assert [result.status for result in report.results] == [
        TrialImportStatus.IMPORTABLE,
        TrialImportStatus.IMPORTABLE,
        TrialImportStatus.IMPORTABLE,
    ]


def test_premise_becomes_stale_after_study_changes() -> None:
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        create_trial(
            params={"x": 1.0}, distributions={"x": FloatDistribution(0, 1)}, value=1.0
        )
    )
    trial = create_trial(
        params={"x": 0.5}, distributions={"x": FloatDistribution(0, 1)}, value=0.5
    )
    report = study.precheck_add_trials(
        [ExternalTrial(trial, directions=[StudyDirection.MINIMIZE])]
    )
    assert report.is_premise_current(study)

    # Another compatible trial changes the observed premise (trial count).
    study.add_trial(
        create_trial(
            params={"x": 0.25, "z": 1},
            distributions={"x": FloatDistribution(0, 1), "z": IntDistribution(0, 10)},
            value=0.25,
        )
    )
    assert report.is_premise_current(study) is False

    # Re-running the precheck binds the report to the new premise.
    new_report = study.precheck_add_trials(
        [ExternalTrial(trial, directions=[StudyDirection.MINIMIZE])]
    )
    assert new_report.is_premise_current(study)
    assert new_report.n_trials == 2


def test_premise_bound_to_study_identity() -> None:
    study = optuna.create_study(direction="minimize")
    other_study = optuna.create_study(direction="minimize")
    trial = create_trial(
        params={"x": 1.0}, distributions={"x": FloatDistribution(0, 1)}, value=1.0
    )
    report = study.precheck_add_trials(
        [ExternalTrial(trial, directions=[StudyDirection.MINIMIZE])]
    )
    assert report.is_premise_current(study)
    assert report.is_premise_current(other_study) is False


def test_actual_import_revalidates_independently() -> None:
    study = optuna.create_study(direction="minimize")
    valid = create_trial(
        params={"x": 1.0}, distributions={"x": FloatDistribution(0, 1)}, value=1.0
    )
    invalid = _complete_trial(
        values=[5.0],
        params={"x": 5.0},
        distributions={"x": FloatDistribution(0, 1)},
    )
    report = study.precheck_add_trials(
        [
            ExternalTrial(valid, directions=[StudyDirection.MINIMIZE]),
            ExternalTrial(invalid, directions=[StudyDirection.MINIMIZE]),
        ]
    )
    assert report.results[0].status == TrialImportStatus.IMPORTABLE
    assert report.results[1].status == TrialImportStatus.CONFLICT
    assert len(study.trials) == 0

    # add_trial performs its own validation regardless of any report.
    study.add_trial(valid)
    with pytest.raises(ValueError):
        study.add_trial(invalid)
    assert len(study.trials) == 1


def test_invalid_record_types_raise() -> None:
    study = optuna.create_study(direction="minimize")
    with pytest.raises(TypeError):
        study.precheck_add_trials(["not-a-trial"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        study.precheck_add_trials(
            [ExternalTrial("not-a-trial")]  # type: ignore[arg-type]
        )


def test_generator_input_and_empty_batch() -> None:
    study = optuna.create_study(direction="minimize")
    empty_report = study.precheck_add_trials(())
    assert empty_report.results == ()
    assert empty_report.all_importable
    assert empty_report.is_premise_current(study)

    def gen():
        yield ExternalTrial(
            create_trial(
                params={"x": 1.0},
                distributions={"x": FloatDistribution(0, 1)},
                value=1.0,
            ),
            directions=[StudyDirection.MINIMIZE],
        )

    report = study.precheck_add_trials(gen())
    assert len(report.results) == 1
    assert report.results[0].index == 0
    assert report.results[0].status == TrialImportStatus.IMPORTABLE


def test_multi_objective_full_roundtrip() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    trial = create_trial(
        values=[1.0, 2.0],
        params={"x": 1.0},
        distributions={"x": FloatDistribution(0, 1)},
    )
    report = study.precheck_add_trials(
        [ExternalTrial(trial, directions=[StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])]
    )
    assert report.results[0].status == TrialImportStatus.IMPORTABLE
    assert report.directions == (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE)

    study.add_trial(trial)
    assert len(study.trials) == 1
