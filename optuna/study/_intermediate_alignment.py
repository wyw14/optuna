from __future__ import annotations

from collections.abc import Container
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from optuna._experimental import experimental_func
from optuna.trial._state import TrialState


if TYPE_CHECKING:
    from optuna.study.study import Study
    from optuna.trial import FrozenTrial


__all__ = [
    "AlignedIntermediateValue",
    "AlignedIntermediateValues",
    "AlignedTrialIntermediateValues",
    "AlignedValueType",
    "align_intermediate_values",
]


# Only piecewise linear interpolation between adjacent reported points is currently supported.
_SUPPORTED_INTERPOLATIONS: tuple[str, ...] = ("linear",)

# States that may have reported intermediate values. Unlike ``plot_intermediate_values``,
# ``FAIL`` trials are included because the values they reported before failing are real data.
_DEFAULT_STATES: tuple[TrialState, ...] = (
    TrialState.RUNNING,
    TrialState.COMPLETE,
    TrialState.PRUNED,
    TrialState.FAIL,
)

# Label assigned to trials that are not explicitly assigned a budget semantics. It denotes
# Optuna's unitless integer ``step`` passed to ``trial.report``.
_DEFAULT_BUDGET_SEMANTICS = "step"


class AlignedValueType(Enum):
    """Provenance of a value in the aligned output.

    Attributes:
        REPORTED:
            The point coincides with a step at which the trial actually reported an
            intermediate value. The value is exactly the reported value.
        INTERPOLATED:
            The point lies strictly between two adjacent reported points. The value was
            derived by the explicit interpolation rule chosen for the alignment.
        UNAVAILABLE:
            The point cannot be derived without extrapolation (it is located before the
            trial's first reported point or after its last reported point). No value is
            given, i.e., it is never filled with zero.
    """

    REPORTED = 1
    INTERPOLATED = 2
    UNAVAILABLE = 3


@dataclass(frozen=True)
class AlignedIntermediateValue:
    """An intermediate value of a trial aligned to a target resource budget point.

    Attributes:
        step:
            The target resource budget point (an integer ``step``).
        value:
            The aligned objective value. It is exactly the reported value when
            :attr:`kind` is :class:`~optuna.study.AlignedValueType.REPORTED`, the value
            produced by the interpolation rule when
            :attr:`kind` is :class:`~optuna.study.AlignedValueType.INTERPOLATED`, and
            :obj:`None` when :attr:`kind` is
            :class:`~optuna.study.AlignedValueType.UNAVAILABLE`.
        kind:
            Provenance of :attr:`value` (reported, interpolated, or unavailable).
        interpolation:
            Name of the interpolation rule used when :attr:`kind` is
            :class:`~optuna.study.AlignedValueType.INTERPOLATED`, otherwise :obj:`None`.
    """

    step: int
    value: float | None
    kind: AlignedValueType
    interpolation: str | None


@dataclass(frozen=True)
class AlignedTrialIntermediateValues:
    """Aligned intermediate values of a single trial.

    The points are ordered by :attr:`~optuna.study.AlignedIntermediateValue.step` and align
    one-to-one with the target budget points of the enclosing alignment result. Failed and
    pruned trials retain the values they actually reported; budget points beyond their last
    reported point are marked
    :class:`~optuna.study.AlignedValueType.UNAVAILABLE` instead of being extrapolated or
    filled with zero.

    Attributes:
        trial_number:
            Number of the trial.
        state:
            State of the trial.
        first_step:
            First step at which the trial reported a value.
        last_step:
            Last step at which the trial reported a value.
        points:
            Aligned points, in the same order as the target budget points.
    """

    trial_number: int
    state: TrialState
    first_step: int
    last_step: int
    points: tuple[AlignedIntermediateValue, ...]

    @property
    def values(self) -> tuple[float | None, ...]:
        """Aligned values in target-point order; missing points are :obj:`None`, never zero."""

        return tuple(p.value for p in self.points)

    @property
    def kinds(self) -> tuple[AlignedValueType, ...]:
        """Provenance of each aligned point in target-point order."""

        return tuple(p.kind for p in self.points)


@dataclass(frozen=True)
class AlignedIntermediateValues:
    """Read-only result of aligning trials' intermediate values to common budget points.

    This object never holds a reference that can mutate the study, and creating it does not
    write anything back to the storage. Interpolated values are therefore not added to any
    trial's :attr:`~optuna.trial.FrozenTrial.intermediate_values` and do not affect pruning
    decisions.

    Attributes:
        target_steps:
            The common, strictly increasing sequence of resource budget points used for the
            alignment.
        interpolation:
            The interpolation rule applied to points located between adjacent reported points.
        budget_semantics:
            The common resource budget semantics of the aligned trials, or :obj:`None` when
            no semantics were specified.
        trials:
            Aligned intermediate values of the trials that reported at least one value.
    """

    target_steps: tuple[int, ...]
    interpolation: str
    budget_semantics: str | None
    trials: tuple[AlignedTrialIntermediateValues, ...]


def _is_strict_int(x: object) -> bool:
    # ``bool`` is a subclass of ``int`` but is not a valid step.
    return isinstance(x, int) and not isinstance(x, bool)


def _validate_budget_range(
    budget_range: tuple[int | None, int | None] | None,
) -> tuple[int | None, int | None]:
    if budget_range is None:
        return None, None

    if (
        not isinstance(budget_range, tuple)
        or len(budget_range) != 2
        or not all(v is None or _is_strict_int(v) for v in budget_range)
    ):
        raise ValueError(
            f"`budget_range` must be a tuple of two integers or Nones, but got {budget_range}."
        )

    low, high = budget_range
    if low is not None and high is not None and low > high:
        raise ValueError(
            f"`budget_range` lower bound must not exceed the upper bound, but got {budget_range}."
        )
    return low, high


def _validate_target_steps(target_steps: Sequence[int]) -> tuple[int, ...]:
    steps = tuple(target_steps)
    if not all(_is_strict_int(s) for s in steps):
        raise ValueError(f"`target_steps` must contain integer steps only, but got {steps}.")
    if any(steps[i] >= steps[i + 1] for i in range(len(steps) - 1)):
        raise ValueError(f"`target_steps` must be strictly increasing, but got {steps}.")
    return steps


def _linear_interpolate(
    target_step: int, left: tuple[int, float], right: tuple[int, float]
) -> float:
    (left_step, left_value), (right_step, right_value) = left, right
    ratio = (target_step - left_step) / (right_step - left_step)
    return left_value + (right_value - left_value) * ratio


def _align_one_trial(
    trial: FrozenTrial, target_steps: tuple[int, ...], interpolation: str
) -> AlignedTrialIntermediateValues:
    reports = sorted(trial.intermediate_values.items())
    first_step = reports[0][0]
    last_step = reports[-1][0]

    points: list[AlignedIntermediateValue] = []
    report_index = 0
    n_reports = len(reports)
    for step in target_steps:
        # Advance to the latest reported step that does not exceed the target step.
        while report_index + 1 < n_reports and reports[report_index + 1][0] <= step:
            report_index += 1

        left_step, left_value = reports[report_index]
        if step == left_step:
            kind = AlignedValueType.REPORTED
            value: float | None = left_value
            rule: str | None = None
        elif step < first_step or step > last_step:
            # No extrapolation in either direction, in particular beyond the last report.
            kind = AlignedValueType.UNAVAILABLE
            value = None
            rule = None
        else:
            # ``step`` is strictly bracketed by two adjacent, real reported points.
            right_step, right_value = reports[report_index + 1]
            kind = AlignedValueType.INTERPOLATED
            if interpolation == "linear":
                value = _linear_interpolate(
                    step, (left_step, left_value), (right_step, right_value)
                )
            else:  # pragma: no cover - guarded by validation in the caller.
                raise ValueError(f"Unsupported interpolation rule: {interpolation}.")
            rule = interpolation

        points.append(
            AlignedIntermediateValue(step=step, value=value, kind=kind, interpolation=rule)
        )

    return AlignedTrialIntermediateValues(
        trial_number=trial.number,
        state=trial.state,
        first_step=first_step,
        last_step=last_step,
        points=tuple(points),
    )


@experimental_func("5.1.0")
def align_intermediate_values(
    study: Study,
    *,
    target_steps: Sequence[int] | None = None,
    budget_range: tuple[int | None, int | None] | None = None,
    interpolation: str = "linear",
    states: Container[TrialState] | None = _DEFAULT_STATES,
    trial_numbers: Container[int] | None = None,
    budget_semantics: Mapping[int, str] | None = None,
) -> AlignedIntermediateValues:
    """Align trials' intermediate values onto a common set of resource budget points.

    Trials often report intermediate values at different strides (e.g., one trial every epoch
    and another every five epochs). This function resamples every trial onto the same target
    budget points by interpolating between adjacent, actually reported points, while keeping
    track of whether each aligned value was reported or produced by an explicit interpolation
    rule.

    Example:

        .. testcode::

            import optuna
            from optuna.trial import create_trial
            from optuna.study import AlignedValueType


            study = optuna.create_study()
            study.add_trial(create_trial(
                value=1.0, intermediate_values={0: 0.0, 1: 1.0, 2: 2.0}
            ))
            study.add_trial(create_trial(
                value=2.0, intermediate_values={0: 0.0, 2: 4.0}
            ))

            aligned = study.align_intermediate_values(target_steps=[0, 1, 2])
            second_trial = aligned.trials[1]
            assert second_trial.points[0].kind == AlignedValueType.REPORTED
            # Step 1 was not reported by the second trial; it is linearly interpolated.
            assert second_trial.points[1].kind == AlignedValueType.INTERPOLATED
            assert second_trial.points[1].value == 2.0
            assert second_trial.points[1].interpolation == "linear"

    The alignment is strictly read-only:

    * No value is extrapolated beyond a trial's first or last reported point. Points that
      cannot be derived are marked
      :class:`~optuna.study.AlignedValueType.UNAVAILABLE` with value :obj:`None`; they are
      never filled with zero.
    * Failed and pruned trials keep the values they reported before they stopped.
    * Nothing is written back to the trials or the storage, so pruning decisions are not
      affected.

    Args:
        study:
            A :class:`~optuna.study.Study` whose trials are aligned.
        target_steps:
            Common resource budget points to align every trial onto. Must be strictly
            increasing integers. If :obj:`None`, the sorted union of all steps reported by
            the selected trials is used.
        budget_range:
            Optional inclusive ``(low, high)`` range used to restrict the target budget
            points; either bound may be :obj:`None`. It only filters the target points and
            never causes extrapolation.
        interpolation:
            Explicit interpolation rule applied between adjacent reported points. Currently
            only ``"linear"`` is supported.
        states:
            Trial states to include. Defaults to running, complete, pruned, and failed
            trials. Set to :obj:`None` to include every state.
        trial_numbers:
            Optional subset of trial numbers to align. Trials outside the subset are
            ignored. Use it to compare groups of trials separately.
        budget_semantics:
            Optional mapping from trial number to the name of the resource budget that the
            trial's steps represent (e.g., ``"epoch"`` or ``"iteration"``). Trials not in
            the mapping are assumed to use Optuna's unitless integer step (``"step"``). When
            the selected trials resolve to more than one semantics, a :class:`ValueError` is
            raised because such learning curves must be compared separately.

    Returns:
        A read-only :class:`~optuna.study.AlignedIntermediateValues` object.
    """

    if interpolation not in _SUPPORTED_INTERPOLATIONS:
        raise ValueError(
            f"Unsupported interpolation rule {interpolation!r}. "
            f"Supported rules are: {', '.join(_SUPPORTED_INTERPOLATIONS)}."
        )

    low, high = _validate_budget_range(budget_range)

    trials = study.get_trials(deepcopy=False, states=states)
    if trial_numbers is not None:
        trials = [t for t in trials if t.number in trial_numbers]
    # Trials that never reported have no learning curve to align.
    trials = [t for t in trials if len(t.intermediate_values) > 0]

    resolved_semantics: str | None
    if budget_semantics is None:
        resolved_semantics = None
    else:
        labels = {
            budget_semantics.get(trial.number, _DEFAULT_BUDGET_SEMANTICS) for trial in trials
        }
        if len(labels) > 1:
            groups = sorted(labels)
            raise ValueError(
                "The selected trials use incompatible resource budget semantics: "
                f"{groups}. Steps with different meanings cannot be aligned onto a common "
                "budget axis; please compare them separately, e.g. by filtering trials with "
                "the `trial_numbers` argument."
            )
        resolved_semantics = next(iter(labels)) if labels else None

    if target_steps is None:
        resolved_target_steps = tuple(
            sorted({step for trial in trials for step in trial.intermediate_values})
        )
    else:
        resolved_target_steps = _validate_target_steps(target_steps)

    if low is not None:
        resolved_target_steps = tuple(s for s in resolved_target_steps if s >= low)
    if high is not None:
        resolved_target_steps = tuple(s for s in resolved_target_steps if s <= high)

    aligned_trials = tuple(
        _align_one_trial(trial, resolved_target_steps, interpolation) for trial in trials
    )

    return AlignedIntermediateValues(
        target_steps=resolved_target_steps,
        interpolation=interpolation,
        budget_semantics=resolved_semantics,
        trials=aligned_trials,
    )
