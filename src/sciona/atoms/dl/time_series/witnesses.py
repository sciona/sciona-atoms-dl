"""Ghost witnesses for time-series forecasting atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_exponential_smoothing_level(
    series: AbstractArray,
    alpha: float,
) -> AbstractArray:
    """Witness for exponential-smoothing level extraction."""
    return series


def witness_multiplicative_seasonality_decompose(
    series: AbstractArray,
    level: AbstractArray,
    season_length: int,
) -> AbstractArray:
    """Witness for multiplicative seasonal factor extraction."""
    return series


def witness_smyl_loss(
    forecast: AbstractArray,
    actual: AbstractArray,
    naive_mae: float,
) -> float:
    """Witness for the combined M4-style scalar loss."""
    return 0.0


def witness_pinball_loss(
    forecast: AbstractArray,
    actual: AbstractArray,
    tau: float,
) -> float:
    """Witness for quantile pinball loss."""
    return 0.0
