from __future__ import annotations

import numpy as np
import pytest


def test_time_series_import() -> None:
    from sciona.atoms.dl.time_series.atoms import (
        exponential_smoothing_level,
        multiplicative_seasonality_decompose,
        pinball_loss,
        smyl_loss,
    )

    assert callable(exponential_smoothing_level)
    assert callable(multiplicative_seasonality_decompose)
    assert callable(smyl_loss)
    assert callable(pinball_loss)


def test_exponential_smoothing_level_basic() -> None:
    from sciona.atoms.dl.time_series.atoms import exponential_smoothing_level

    series = np.array([10.0, 12.0, 14.0], dtype=np.float64)
    result = exponential_smoothing_level(series, alpha=0.5)
    expected = np.array([10.0, 11.0, 12.5], dtype=np.float64)
    np.testing.assert_allclose(result, expected)


def test_exponential_smoothing_level_alpha_one_tracks_series() -> None:
    from sciona.atoms.dl.time_series.atoms import exponential_smoothing_level

    series = np.array([3.0, 5.0, 7.0], dtype=np.float64)
    result = exponential_smoothing_level(series, alpha=1.0)
    np.testing.assert_allclose(result, series)


def test_multiplicative_seasonality_decompose_pointwise_ratio() -> None:
    from sciona.atoms.dl.time_series.atoms import multiplicative_seasonality_decompose

    series = np.array([10.0, 12.0, 11.0, 13.0], dtype=np.float64)
    level = np.array([10.0, 11.0, 11.0, 12.0], dtype=np.float64)
    result = multiplicative_seasonality_decompose(series, level, season_length=2)
    expected = np.array([1.0, 12.0 / 11.0, 1.0, 13.0 / 12.0], dtype=np.float64)
    np.testing.assert_allclose(result, expected)


def test_pinball_loss_matches_quantile_formula() -> None:
    from sciona.atoms.dl.time_series.atoms import pinball_loss

    forecast = np.array([0.0, 3.0], dtype=np.float64)
    actual = np.array([1.0, 2.0], dtype=np.float64)
    result = pinball_loss(forecast, actual, tau=0.5)
    assert result == 0.5


def test_pinball_loss_respects_tau_asymmetry() -> None:
    from sciona.atoms.dl.time_series.atoms import pinball_loss

    forecast = np.array([0.0, 2.0], dtype=np.float64)
    actual = np.array([1.0, 0.0], dtype=np.float64)
    result = pinball_loss(forecast, actual, tau=0.8)
    expected = np.mean([0.8, 0.4])
    assert result == pytest.approx(expected)


def test_smyl_loss_zero_for_perfect_forecast() -> None:
    from sciona.atoms.dl.time_series.atoms import smyl_loss

    forecast = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    actual = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert smyl_loss(forecast, actual, naive_mae=2.0) == 0.0


def test_smyl_loss_averages_smape_and_mase() -> None:
    from sciona.atoms.dl.time_series.atoms import smyl_loss

    forecast = np.array([8.0, 12.0], dtype=np.float64)
    actual = np.array([10.0, 10.0], dtype=np.float64)
    result = smyl_loss(forecast, actual, naive_mae=4.0)

    abs_error = np.abs(actual - forecast)
    smape = np.mean(2.0 * abs_error / (np.abs(actual) + np.abs(forecast)))
    mase = np.mean(abs_error) / 4.0
    expected = 0.5 * (smape + mase)
    assert result == pytest.approx(expected)


def test_smyl_loss_handles_zero_zero_smape_terms() -> None:
    from sciona.atoms.dl.time_series.atoms import smyl_loss

    forecast = np.array([0.0, 1.0], dtype=np.float64)
    actual = np.array([0.0, 2.0], dtype=np.float64)
    result = smyl_loss(forecast, actual, naive_mae=1.0)
    assert np.isfinite(result)
