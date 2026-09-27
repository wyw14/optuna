from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import enum
import math
from typing import TYPE_CHECKING

from optuna.distributions import _convert_old_distribution_to_new_distribution
from optuna.distributions import BaseDistribution
from optuna.distributions import check_distribution_compatibility
from optuna.study._study_direction import StudyDirection
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


if TYPE_CHECKING:
    from optuna.study.study import Study


# --------------------------------------------------------------------------------------
# Public result types
# --------------------------------------------------------------------------------------


class TrialImportStatus(str, enum.Enum):
    """Per-record verdict of an import precheck.

    Attributes:
        IMPORTABLE:
            The record satisfies every requirement that can be checked and can be imported.
        CONFLICT:
            The record violates a concrete requirement: it is invalid on its own, conflicts
            with the observed target study, or is inconsistent with another record in the same
            batch. See :attr:`PrecheckIssue.category` to distinguish these cases.
        UNCONFIRMED:
            No concrete violation was found, but a requirement cannot be verified from the
            available information (for example because the record carries no source-study
            direction). Importing the record is not blocked, but the caller should confirm the
            missing premise manually.
    """

    IMPORTABLE = "importable"
    CONFLICT = "conflict"
    UNCONFIRMED = "unconfirmed"


class PrecheckIssueCategory(str, enum.Enum):
    """Category of a precheck issue.

    Attributes:
        SELF:
            The record is invalid on its own. It would fail
            :func:`~optuna.trial.FrozenTrial._validate` even before the target study is
            considered.
        TARGET_STUDY:
            The record conflicts with the target study premise observed during the precheck
            (number of objectives, directions, or the already recorded parameter space).
        BATCH:
            The record is not invalid on its own, but it is inconsistent with an earlier record
            in the same batch, so importing the batch as a whole would fail.
        UNVERIFIED:
            A requirement cannot be confirmed or denied from the record and the observed target
            study.
    """

    SELF = "self"
    TARGET_STUDY = "target_study"
    BATCH = "batch"
    UNVERIFIED = "unverified"


@dataclasses.dataclass(frozen=True)
class PrecheckIssue:
    """A single problem (or unverifiable point) reported for one record.

    Attributes:
        code:
            Machine-readable reason code.
        category:
            Whether the issue is caused by the record itself, by the target study, by another
            record in the same batch, or by missing information.
        message:
            Human-readable explanation.
        param_name:
            Name of the parameter involved, if the issue is parameter specific.
        objective:
            Zero-based index of the objective involved, if the issue is objective specific.
        conflicting_index:
            For a :attr:`PrecheckIssueCategory.BATCH` issue, the index of the earlier record in
            the same batch that the current record conflicts with.
    """

    code: str
    category: PrecheckIssueCategory
    message: str
    param_name: str | None = None
    objective: int | None = None
    conflicting_index: int | None = None


@dataclasses.dataclass(frozen=True)
class ExternalTrial:
    """An external trial record with optional provenance of its source study.

    A bare :class:`~optuna.trial.FrozenTrial` does not carry the optimization direction of the
    study it originated from. Wrap it in this class and provide ``directions`` so that the
    precheck can confirm direction consistency with the target study.

    Attributes:
        trial:
            The external trial to be imported.
        directions:
            Directions of the source study that the trial originated from. :obj:`None` means the
            provenance is unknown, in which case direction consistency cannot be confirmed.
    """

    trial: FrozenTrial
    directions: Sequence[StudyDirection] | None = None


@dataclasses.dataclass(frozen=True)
class TrialPrecheckResult:
    """Precheck verdict for a single record.

    Attributes:
        index:
            Position of the record in the input batch (zero based).
        status:
            Verdict of the record.
        record:
            The checked :class:`~optuna.trial.FrozenTrial`.
        source_directions:
            Source-study directions attached to the record, if any.
        issues:
            All issues found for the record. The record is importable iff this is empty.
    """

    index: int
    status: TrialImportStatus
    record: FrozenTrial
    source_directions: tuple[StudyDirection, ...] | None
    issues: tuple[PrecheckIssue, ...]

    @property
    def importable(self) -> bool:
        """:obj:`True` iff the record has no issue (including unverifiable ones)."""

        return self.status == TrialImportStatus.IMPORTABLE


@dataclasses.dataclass(frozen=True)
class StudyPrecheckReport:
    """Result of a read-only import precheck, bound to the observed target study.

    The report is only an advice. It is never written to storage and cannot be passed to
    :func:`~optuna.study.Study.add_trial` or :func:`~optuna.study.Study.add_trials`; those
    methods always perform their own validation. Use :meth:`is_premise_current` to detect
    whether the target study has changed since the report was produced; if it has, run the
    precheck again instead of trusting the old report.

    Attributes:
        study_name:
            Name of the observed target study.
        directions:
            Directions of the target study observed during the precheck.
        distributions:
            Parameter space of the target study observed during the precheck, mapping each
            parameter name to a representative recorded distribution. Note that numerical
            (:class:`~optuna.distributions.FloatDistribution` /
            :class:`~optuna.distributions.IntDistribution`) bounds can legitimately differ
            between trials (dynamic parameter space); only the distribution kind, the ``log``
            configuration and categorical choices are fixed.
        n_trials:
            Number of trials in the target study observed during the precheck.
        results:
            Per-record verdicts, in input order.
    """

    study_name: str
    directions: tuple[StudyDirection, ...]
    distributions: Mapping[str, BaseDistribution]
    n_trials: int
    results: tuple[TrialPrecheckResult, ...]

    @property
    def all_importable(self) -> bool:
        """:obj:`True` iff every record in the batch has status IMPORTABLE."""

        return all(result.status == TrialImportStatus.IMPORTABLE for result in self.results)

    @property
    def importable_indices(self) -> tuple[int, ...]:
        """Indices of records with status IMPORTABLE."""

        return tuple(
            result.index
            for result in self.results
            if result.status == TrialImportStatus.IMPORTABLE
        )

    @property
    def blocked_indices(self) -> tuple[int, ...]:
        """Indices of records with status CONFLICT."""

        return tuple(
            result.index for result in self.results if result.status == TrialImportStatus.CONFLICT
        )

    @property
    def unconfirmed_indices(self) -> tuple[int, ...]:
        """Indices of records with status UNCONFIRMED."""

        return tuple(
            result.index
            for result in self.results
            if result.status == TrialImportStatus.UNCONFIRMED
        )

    def is_premise_current(self, study: Study) -> bool:
        """Check whether the observed target-study premise still holds.

        This re-reads the target study and compares it with the information bound to this
        report. :obj:`False` means the directions, the recorded parameter space, the number of
        trials, or the study identity changed since the precheck; the report must not be trusted
        anymore and the precheck (or the actual import validation) should run again.

        Args:
            study:
                The target study to compare against. It should be the same study that produced
                this report.
        """

        if study.study_name != self.study_name:
            return False

        storage = study._storage
        study_id = study._study_id
        if tuple(storage.get_study_directions(study_id)) != self.directions:
            return False

        current_trials = storage.get_all_trials(study_id, deepcopy=False)
        if len(current_trials) != self.n_trials:
            return False

        current_space = _observe_distribution_space(current_trials)
        if set(current_space.keys()) != set(self.distributions.keys()):
            return False
        for name, current_distribution in current_space.items():
            observed_distribution = self.distributions[name]
            try:
                check_distribution_compatibility(observed_distribution, current_distribution)
                check_distribution_compatibility(current_distribution, observed_distribution)
            except ValueError:
                return False
        return True


# --------------------------------------------------------------------------------------
# Reason codes
# --------------------------------------------------------------------------------------


class _Code:
    # SELF
    INVALID_STATE = "invalid_state"
    MISSING_DATETIME_START = "missing_datetime_start"
    MISSING_DATETIME_COMPLETE = "missing_datetime_complete"
    UNEXPECTED_DATETIME_COMPLETE = "unexpected_datetime_complete"
    FAILED_TRIAL_HAS_VALUES = "failed_trial_has_values"
    COMPLETE_TRIAL_MISSING_VALUES = "complete_trial_missing_values"
    NAN_OBJECTIVE_VALUE = "nan_objective_value"
    INVALID_PARAMS_DISTRIBUTIONS = "invalid_params_distributions"
    PARAMS_DISTRIBUTIONS_KEY_MISMATCH = "params_distributions_key_mismatch"
    INVALID_DISTRIBUTION = "invalid_distribution"
    PARAM_VALUE_OUT_OF_DISTRIBUTION = "param_value_out_of_distribution"
    INVALID_SOURCE_DIRECTIONS = "invalid_source_directions"
    # TARGET_STUDY
    OBJECTIVE_COUNT_MISMATCH = "objective_count_mismatch"
    DIRECTION_MISMATCH = "direction_mismatch"
    DISTRIBUTION_CONFLICT = "distribution_conflict"
    # BATCH
    BATCH_DISTRIBUTION_CONFLICT = "batch_distribution_conflict"
    # UNVERIFIED
    DIRECTION_UNCONFIRMED = "direction_unconfirmed"


_BLOCKING_CATEGORIES = (
    PrecheckIssueCategory.SELF,
    PrecheckIssueCategory.TARGET_STUDY,
    PrecheckIssueCategory.BATCH,
)


# --------------------------------------------------------------------------------------
# Core logic
# --------------------------------------------------------------------------------------


def precheck_add_trials(
    study: Study,
    records: Iterable[FrozenTrial | ExternalTrial],
) -> StudyPrecheckReport:
    """Check external trials against a target study without writing anything to storage.

    Each record is classified as importable, conflicting, or unconfirmable. Issues are split
    into records that are invalid on their own and records that are merely inconsistent with
    each other in the same batch. Numerical distribution bounds are treated as a dynamic
    parameter space: widening or narrowing ``low`` / ``high`` / ``step`` is accepted as long as
    the distribution kind and the ``log`` configuration match; categorical choices must match
    exactly.

    This function is read-only. It does not create trials and it does not modify the study. The
    returned report is bound to the observed target study premise and cannot be used to bypass
    validation when the trials are actually added via
    :func:`~optuna.study.Study.add_trial`.

    Args:
        study:
            The target study that the trials would be added to.
        records:
            External trials to check. Each item is either a
            :class:`~optuna.trial.FrozenTrial` or an :class:`ExternalTrial` carrying the
            directions of its source study. A bare :class:`~optuna.trial.FrozenTrial` cannot be
            checked for direction consistency and is reported as UNCONFIRMED when it carries
            objective values.

    Returns:
        A :class:`StudyPrecheckReport` containing per-record verdicts and the observed target
        study premise.
    """

    normalized: list[tuple[FrozenTrial, tuple[StudyDirection, ...] | None]] = []
    for index, record in enumerate(records):
        if isinstance(record, ExternalTrial):
            if not isinstance(record.trial, FrozenTrial):
                raise TypeError(
                    "Each ExternalTrial.trial must be a FrozenTrial, but the record at index "
                    f"{index} has {type(record.trial).__name__}."
                )
            directions = (
                tuple(record.directions) if record.directions is not None else None
            )
            normalized.append((record.trial, directions))
        elif isinstance(record, FrozenTrial):
            normalized.append((record, None))
        else:
            raise TypeError(
                "Each record must be a FrozenTrial or an ExternalTrial, but the record at "
                f"index {index} is of type {type(record).__name__}."
            )

    storage = study._storage
    study_id = study._study_id
    target_directions = tuple(storage.get_study_directions(study_id))
    target_trials = storage.get_all_trials(study_id, deepcopy=False)
    target_space = _observe_distribution_space(target_trials)

    # Simulated space: starts from the observed target space and grows with the records that
    # could actually be added. ``space_origins[name]`` is None for parameters already present in
    # the target study and otherwise the index of the batch record that introduced it.
    simulated_space: dict[str, BaseDistribution] = dict(target_space)
    space_origins: dict[str, int | None] = {name: None for name in target_space}

    results: list[TrialPrecheckResult] = []
    for index, (trial, raw_source_directions) in enumerate(normalized):
        issues: list[PrecheckIssue] = []

        valid_param_names = _check_self(trial, raw_source_directions, issues)
        source_directions = _validated_source_directions(
            trial, raw_source_directions, issues
        )

        _check_objective_count(trial, target_directions, issues)
        _check_directions(trial, source_directions, target_directions, issues)
        _check_and_extend_distribution_space(
            index,
            trial,
            valid_param_names,
            simulated_space,
            space_origins,
            issues,
        )

        if any(issue.category in _BLOCKING_CATEGORIES for issue in issues):
            status = TrialImportStatus.CONFLICT
        elif any(issue.category == PrecheckIssueCategory.UNVERIFIED for issue in issues):
            status = TrialImportStatus.UNCONFIRMED
        else:
            status = TrialImportStatus.IMPORTABLE

        results.append(
            TrialPrecheckResult(
                index=index,
                status=status,
                record=trial,
                source_directions=source_directions,
                issues=tuple(issues),
            )
        )

    return StudyPrecheckReport(
        study_name=study.study_name,
        directions=target_directions,
        distributions=target_space,
        n_trials=len(target_trials),
        results=tuple(results),
    )


def _observe_distribution_space(
    trials: Sequence[FrozenTrial],
) -> dict[str, BaseDistribution]:
    """Collect a representative distribution per parameter name.

    Storage guarantees that recorded distributions sharing a name are mutually compatible, so
    the first occurrence (trials are ordered by number) represents the fixed part of the space.
    """

    space: dict[str, BaseDistribution] = {}
    for trial in trials:
        for name, distribution in trial.distributions.items():
            if name not in space:
                space[name] = _convert_old_distribution_to_new_distribution(distribution)
    return space


def _issue(
    issues: list[PrecheckIssue],
    code: str,
    category: PrecheckIssueCategory,
    message: str,
    *,
    param_name: str | None = None,
    objective: int | None = None,
    conflicting_index: int | None = None,
) -> None:
    issues.append(
        PrecheckIssue(
            code=code,
            category=category,
            message=message,
            param_name=param_name,
            objective=objective,
            conflicting_index=conflicting_index,
        )
    )


def _check_self(
    trial: FrozenTrial,
    raw_source_directions: Sequence[StudyDirection] | None,
    issues: list[PrecheckIssue],
) -> set[str]:
    """Validate the record on its own, mirroring ``FrozenTrial._validate``.

    Returns the names of parameters whose own distribution and value are valid; those are the
    only distributions that may participate in target-study and batch space checks.
    """

    valid_param_names: set[str] = set()

    state = trial.state
    if not isinstance(state, TrialState):
        _issue(
            issues,
            _Code.INVALID_STATE,
            PrecheckIssueCategory.SELF,
            f"Trial state must be a TrialState but got {state!r}.",
        )
    else:
        if state != TrialState.WAITING and trial.datetime_start is None:
            _issue(
                issues,
                _Code.MISSING_DATETIME_START,
                PrecheckIssueCategory.SELF,
                "`datetime_start` is supposed to be set when the trial state is not waiting.",
            )
        if state.is_finished():
            if trial.datetime_complete is None:
                _issue(
                    issues,
                    _Code.MISSING_DATETIME_COMPLETE,
                    PrecheckIssueCategory.SELF,
                    "`datetime_complete` is supposed to be set for a finished trial.",
                )
        elif trial.datetime_complete is not None:
            _issue(
                issues,
                _Code.UNEXPECTED_DATETIME_COMPLETE,
                PrecheckIssueCategory.SELF,
                "`datetime_complete` is supposed to be None for an unfinished trial.",
            )

        if state == TrialState.FAIL and trial.values is not None:
            _issue(
                issues,
                _Code.FAILED_TRIAL_HAS_VALUES,
                PrecheckIssueCategory.SELF,
                f"values should be None for a failed trial, but got {trial.values}.",
            )
        if state == TrialState.COMPLETE:
            if trial.values is None:
                _issue(
                    issues,
                    _Code.COMPLETE_TRIAL_MISSING_VALUES,
                    PrecheckIssueCategory.SELF,
                    "values should be set for a complete trial.",
                )
            elif any(math.isnan(value) for value in trial.values):
                _issue(
                    issues,
                    _Code.NAN_OBJECTIVE_VALUE,
                    PrecheckIssueCategory.SELF,
                    "values should not contain NaN.",
                )

    params = trial.params
    distributions = trial.distributions
    if not isinstance(params, Mapping) or not isinstance(distributions, Mapping):
        _issue(
            issues,
            _Code.INVALID_PARAMS_DISTRIBUTIONS,
            PrecheckIssueCategory.SELF,
            "`params` and `distributions` of the trial must both be dictionaries.",
        )
        return valid_param_names

    if set(params.keys()) != set(distributions.keys()):
        _issue(
            issues,
            _Code.PARAMS_DISTRIBUTIONS_KEY_MISMATCH,
            PrecheckIssueCategory.SELF,
            f"Inconsistent parameters {set(params.keys())} and distributions "
            f"{set(distributions.keys())}.",
        )

    for param_name, param_value in params.items():
        if param_name not in distributions:
            continue
        distribution = distributions[param_name]
        if not isinstance(distribution, BaseDistribution):
            _issue(
                issues,
                _Code.INVALID_DISTRIBUTION,
                PrecheckIssueCategory.SELF,
                f"Distribution of parameter '{param_name}' must be a BaseDistribution instance "
                f"but got {type(distribution).__name__}.",
                param_name=param_name,
            )
            continue
        distribution = _convert_old_distribution_to_new_distribution(distribution)
        try:
            contained = distribution._contains(distribution.to_internal_repr(param_value))
        except (TypeError, ValueError):
            contained = False
        if not contained:
            _issue(
                issues,
                _Code.PARAM_VALUE_OUT_OF_DISTRIBUTION,
                PrecheckIssueCategory.SELF,
                f"The value {param_value!r} of parameter '{param_name}' isn't contained in "
                f"the distribution {distribution}.",
                param_name=param_name,
            )
            continue
        valid_param_names.add(param_name)

    return valid_param_names


def _validated_source_directions(
    trial: FrozenTrial,
    raw_source_directions: Sequence[StudyDirection] | None,
    issues: list[PrecheckIssue],
) -> tuple[StudyDirection, ...] | None:
    if raw_source_directions is None:
        return None

    if not all(isinstance(direction, StudyDirection) for direction in raw_source_directions):
        _issue(
            issues,
            _Code.INVALID_SOURCE_DIRECTIONS,
            PrecheckIssueCategory.SELF,
            "Source directions must be StudyDirection values "
            f"but got {list(raw_source_directions)!r}.",
        )
        return None
    directions = tuple(raw_source_directions)

    if any(direction == StudyDirection.NOT_SET for direction in directions):
        _issue(
            issues,
            _Code.INVALID_SOURCE_DIRECTIONS,
            PrecheckIssueCategory.SELF,
            "Source directions must not contain StudyDirection.NOT_SET but got "
            f"{list(directions)!r}.",
        )
        return None

    if trial.values is not None and len(directions) != len(trial.values):
        _issue(
            issues,
            _Code.INVALID_SOURCE_DIRECTIONS,
            PrecheckIssueCategory.SELF,
            f"The record carries {len(directions)} source directions but "
            f"{len(trial.values)} objective values.",
        )
        return None

    return directions


def _check_objective_count(
    trial: FrozenTrial,
    target_directions: Sequence[StudyDirection],
    issues: list[PrecheckIssue],
) -> None:
    if trial.values is not None and len(trial.values) != len(target_directions):
        _issue(
            issues,
            _Code.OBJECTIVE_COUNT_MISMATCH,
            PrecheckIssueCategory.TARGET_STUDY,
            f"The record has {len(trial.values)} objective values, but the target study has "
            f"{len(target_directions)} objective(s).",
        )


def _check_directions(
    trial: FrozenTrial,
    source_directions: Sequence[StudyDirection] | None,
    target_directions: Sequence[StudyDirection],
    issues: list[PrecheckIssue],
) -> None:
    if trial.values is None:
        return

    if source_directions is None:
        _issue(
            issues,
            _Code.DIRECTION_UNCONFIRMED,
            PrecheckIssueCategory.UNVERIFIED,
            "The record carries no source-study directions, so its optimization direction "
            "cannot be compared with the target study directions "
            f"{[direction.name for direction in target_directions]}.",
        )
        return

    # Length mismatch is reported separately as a self/objective-count issue; comparing
    # overlapping objectives would only add noise.
    if len(source_directions) != len(target_directions):
        return

    for objective, (source_direction, target_direction) in enumerate(
        zip(source_directions, target_directions)
    ):
        if source_direction != target_direction:
            _issue(
                issues,
                _Code.DIRECTION_MISMATCH,
                PrecheckIssueCategory.TARGET_STUDY,
                f"Objective {objective} has direction {source_direction.name} in the record's "
                f"source study but {target_direction.name} in the target study.",
                objective=objective,
            )


def _check_and_extend_distribution_space(
    index: int,
    trial: FrozenTrial,
    valid_param_names: set[str],
    simulated_space: dict[str, BaseDistribution],
    space_origins: dict[str, int | None],
    issues: list[PrecheckIssue],
) -> None:
    distributions = trial.distributions
    if not isinstance(distributions, Mapping):
        return

    for param_name, raw_distribution in distributions.items():
        if param_name not in valid_param_names:
            # The record fails its own validation for this parameter, so it would never reach
            # the storage-level distribution check nor extend the simulated space.
            continue
        distribution = _convert_old_distribution_to_new_distribution(raw_distribution)

        if param_name in simulated_space:
            existing = simulated_space[param_name]
            try:
                check_distribution_compatibility(existing, distribution)
            except ValueError as e:
                origin = space_origins[param_name]
                if origin is None:
                    _issue(
                        issues,
                        _Code.DISTRIBUTION_CONFLICT,
                        PrecheckIssueCategory.TARGET_STUDY,
                        f"Distribution of parameter '{param_name}' conflicts with the parameter "
                        f"space already recorded in the target study: {e}",
                        param_name=param_name,
                    )
                else:
                    _issue(
                        issues,
                        _Code.BATCH_DISTRIBUTION_CONFLICT,
                        PrecheckIssueCategory.BATCH,
                        f"Distribution of parameter '{param_name}' conflicts with the "
                        f"distribution introduced by record {origin} in the same batch: {e}",
                        param_name=param_name,
                        conflicting_index=origin,
                    )
                continue
            # Compatible existing parameter: numerical bounds legitimately evolve (dynamic
            # parameter space), but the introducing origin must not change.
            simulated_space[param_name] = distribution
            continue

        # A new parameter introduced by the current batch record.
        space_origins[param_name] = index
        simulated_space[param_name] = distribution
