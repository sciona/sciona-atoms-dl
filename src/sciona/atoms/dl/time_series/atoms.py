"""Time-series forecasting primitives in pure numpy.

Implements core components extracted from the MIT-licensed ESRNN codebase:
recursive level smoothing, multiplicative seasonal decomposition, and
forecast losses used around the M4 ES-RNN pipeline.

Source: m4-esrnn-1st/ESRNN/utils/ESRNN.py (MIT)
        m4-esrnn-1st/ESRNN/utils/losses.py (MIT)
        m4-esrnn-1st/ESRNN/utils_evaluation.py (MIT, conceptual metric pieces)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_exponential_smoothing_level,
    witness_multiplicative_seasonality_decompose,
    witness_pinball_loss,
    witness_smyl_loss,
)


def _as_float_vector(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return array.reshape(-1)


@register_atom(witness_exponential_smoothing_level)
@icontract.require(lambda series: np.asarray(series).size >= 1, "series must be non-empty")
@icontract.require(lambda alpha: 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]")
@icontract.ensure(
    lambda series, result: result.shape == np.asarray(series).reshape(-1).shape,
    "result must match the series shape",
)
def exponential_smoothing_level(
    series: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Compute the recursive exponential-smoothing level sequence.

    Uses the same multiplicative-ES level update pattern as ESRNN, reduced to
    a single-series numpy primitive:

        level[t] = alpha * series[t] + (1 - alpha) * level[t - 1]

    The initial level is anchored at the first observation.
    """
    x = _as_float_vector(series)
    level = np.empty_like(x)
    level[0] = x[0]
    for t in range(1, len(x)):
        level[t] = alpha * x[t] + (1.0 - alpha) * level[t - 1]
    return level


@register_atom(witness_multiplicative_seasonality_decompose)
@icontract.require(lambda series: np.asarray(series).size >= 1, "series must be non-empty")
@icontract.require(lambda level: np.asarray(level).size >= 1, "level must be non-empty")
@icontract.require(
    lambda series, level: np.asarray(series).reshape(-1).shape
    == np.asarray(level).reshape(-1).shape,
    "series and level must have the same shape",
)
@icontract.require(lambda season_length: season_length >= 1, "season_length must be positive")
@icontract.require(
    lambda series, season_length: season_length <= np.asarray(series).size,
    "season_length must not exceed the series length",
)
@icontract.require(
    lambda level: np.all(np.asarray(level, dtype=np.float64) != 0.0),
    "level must be non-zero for multiplicative decomposition",
)
@icontract.ensure(
    lambda series, result: result.shape == np.asarray(series).reshape(-1).shape,
    "result must match the series shape",
)
def multiplicative_seasonality_decompose(
    series: NDArray[np.float64],
    level: NDArray[np.float64],
    season_length: int,
) -> NDArray[np.float64]:
    """Extract multiplicative seasonal factors as observation-to-level ratios.

    The returned factor at each time step is the local multiplicative
    deseasonalization term:

        seasonal[t] = series[t] / level[t]

    ``season_length`` is retained in the signature because ESRNN indexes
    seasonal states by a known period, even though this atom returns the
    direct factor sequence rather than a compressed seasonal template.
    """
    _ = season_length
    x = _as_float_vector(series)
    lev = _as_float_vector(level)
    return x / lev


@register_atom(witness_smyl_loss)
@icontract.require(
    lambda forecast, actual: np.asarray(forecast).reshape(-1).shape
    == np.asarray(actual).reshape(-1).shape,
    "forecast and actual must have the same shape",
)
@icontract.require(
    lambda forecast: np.asarray(forecast).size >= 1,
    "forecast must be non-empty",
)
@icontract.require(lambda naive_mae: naive_mae > 0.0, "naive_mae must be positive")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def smyl_loss(
    forecast: NDArray[np.float64],
    actual: NDArray[np.float64],
    naive_mae: float,
) -> float:
    """Combine M4-style sMAPE and MASE into a scalar loss.

    Note that the upstream ``SmylLoss`` class in ``utils/losses.py`` is the
    training loss used inside ESRNN (pinball plus optional level smoothness).
    The prompt for this repo requests an evaluation-style scalar with the
    signature ``(forecast, actual, naive_mae) -> float`` instead, so this atom
    composes the M4 metric pieces implemented in ``utils_evaluation.py``:

      1. sMAPE = mean(2 * |actual - forecast| / (|actual| + |forecast|))
      2. MASE = mean(|actual - forecast|) / naive_mae

    The two scale-free terms are averaged into a single score.
    """
    y_hat = _as_float_vector(forecast)
    y = _as_float_vector(actual)

    abs_error = np.abs(y - y_hat)
    denominator = np.abs(y) + np.abs(y_hat)
    smape_terms = np.divide(
        2.0 * abs_error,
        denominator,
        out=np.zeros_like(abs_error),
        where=denominator > 0.0,
    )
    smape = float(np.mean(smape_terms))
    mase = float(np.mean(abs_error) / naive_mae)
    return 0.5 * (smape + mase)


@register_atom(witness_pinball_loss)
@icontract.require(
    lambda forecast, actual: np.asarray(forecast).reshape(-1).shape
    == np.asarray(actual).reshape(-1).shape,
    "forecast and actual must have the same shape",
)
@icontract.require(
    lambda forecast: np.asarray(forecast).size >= 1,
    "forecast must be non-empty",
)
@icontract.require(lambda tau: 0.0 <= tau <= 1.0, "tau must be in [0, 1]")
@icontract.ensure(lambda result: result >= 0.0, "loss must be non-negative")
def pinball_loss(
    forecast: NDArray[np.float64],
    actual: NDArray[np.float64],
    tau: float,
) -> float:
    """Compute the quantile-regression pinball loss.

    Mirrors the ESRNN ``PinballLoss`` implementation:

        delta = actual - forecast
        loss = mean(max(tau * delta, (tau - 1) * delta))
    """
    y_hat = _as_float_vector(forecast)
    y = _as_float_vector(actual)
    delta = y - y_hat
    pinball = np.maximum(tau * delta, (tau - 1.0) * delta)
    return float(np.mean(pinball))
