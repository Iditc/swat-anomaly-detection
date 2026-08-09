"""Unit tests for preprocessing and feature engineering functions."""

import pandas as pd
import pytest

from src.preprocessing.feature_engineering import (
    compute_constant_sensor_changes,
    compute_contradiction_features,
    compute_deviation,
    compute_frozen_detection,
    compute_pair_residuals,
    compute_rate_of_change,
    compute_rolling_features,
    compute_spike_detection,
)


@pytest.fixture
def sample_series():
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="LIT101")


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "FIT101": [2.5, 2.6, 2.5, 2.7, 2.5],
        "LIT101": [500.0, 500.1, 500.2, 500.3, 500.4],
        "P101": [1, 1, 1, 1, 1],
        "MV101": [1, 1, 0, 0, 1],
    })


class TestRollingFeatures:
    def test_rolling_mean(self, sample_series):
        result = compute_rolling_features(
            sample_series, "LIT101", [3], ["rolling_mean"],
        )
        assert "LIT101_mean_3" in result
        assert len(result["LIT101_mean_3"]) == 5
        assert result["LIT101_mean_3"].iloc[4] == pytest.approx(4.0)

    def test_rolling_std(self, sample_series):
        result = compute_rolling_features(
            sample_series, "LIT101", [3], ["rolling_std"],
        )
        assert "LIT101_std_3" in result
        assert result["LIT101_std_3"].iloc[0] == 0  # single value → fillna(0)

    def test_multiple_windows(self, sample_series):
        result = compute_rolling_features(
            sample_series, "LIT101", [3, 5], ["rolling_mean", "rolling_std"],
        )
        assert len(result) == 4
        assert "LIT101_mean_3" in result
        assert "LIT101_mean_5" in result
        assert "LIT101_std_3" in result
        assert "LIT101_std_5" in result

    def test_empty_features_list(self, sample_series):
        result = compute_rolling_features(
            sample_series, "LIT101", [3], [],
        )
        assert len(result) == 0


class TestRateOfChange:
    def test_basic(self, sample_series):
        result = compute_rate_of_change(sample_series, "LIT101")
        assert "LIT101_roc" in result
        assert result["LIT101_roc"].iloc[0] == 0  # first value fillna(0)
        assert result["LIT101_roc"].iloc[1] == pytest.approx(1.0)

    def test_constant_series(self):
        series = pd.Series([5.0, 5.0, 5.0])
        result = compute_rate_of_change(series, "AIT201")
        assert all(result["AIT201_roc"].iloc[1:] == 0)


class TestDeviation:
    def test_with_nonzero_std(self, sample_series):
        result = compute_deviation(sample_series, "LIT101", mean=3.0, std=1.0)
        assert "LIT101_dev" in result
        assert result["LIT101_dev"].iloc[0] == pytest.approx(2.0)
        assert result["LIT101_dev"].iloc[2] == pytest.approx(0.0)

    def test_with_zero_std(self, sample_series):
        result = compute_deviation(sample_series, "LIT101", mean=3.0, std=0.0)
        assert "LIT101_dev" in result
        assert result["LIT101_dev"].iloc[0] == pytest.approx(2.0)


class TestPairResiduals:
    def test_basic(self, sample_df):
        pairs = [{"sensor_a": "FIT101", "sensor_b": "LIT101"}]
        result = compute_pair_residuals(sample_df, pairs)
        assert "FIT101_LIT101_residual" in result
        assert all(result["FIT101_LIT101_residual"] >= 0)

    def test_missing_sensor(self, sample_df):
        pairs = [{"sensor_a": "FIT101", "sensor_b": "MISSING"}]
        result = compute_pair_residuals(sample_df, pairs)
        assert len(result) == 0

    def test_empty_pairs(self, sample_df):
        result = compute_pair_residuals(sample_df, [])
        assert len(result) == 0


class TestFrozenDetection:
    def test_frozen_sensor(self):
        df = pd.DataFrame({"LIT101": [5.0] * 20})
        result = compute_frozen_detection(df, ["LIT101"], window=10)
        assert "LIT101_frozen_10" in result
        assert result["LIT101_frozen_10"].iloc[-1] == 1

    def test_varying_sensor(self):
        df = pd.DataFrame({"LIT101": list(range(20))})
        result = compute_frozen_detection(df, ["LIT101"], window=10)
        assert result["LIT101_frozen_10"].iloc[-1] == 0


class TestSpikeDetection:
    def test_no_spike(self):
        df = pd.DataFrame({"FIT101": [2.5] * 30})
        result = compute_spike_detection(df, ["FIT101"], window=30)
        assert "FIT101_spike_30" in result
        assert result["FIT101_spike_30"].iloc[-1] == pytest.approx(1.0)

    def test_with_spike(self):
        values = [2.5] * 29 + [100.0]
        df = pd.DataFrame({"FIT101": values})
        result = compute_spike_detection(df, ["FIT101"], window=30)
        assert result["FIT101_spike_30"].iloc[-1] > 5.0


class TestConstantSensorChanges:
    def test_no_change(self):
        df = pd.DataFrame({"P205": [0, 0, 0, 0, 0]})
        result = compute_constant_sensor_changes(df, ["P205"])
        assert "P205_changed" in result
        assert result["P205_changed"].sum() == 0

    def test_with_change(self):
        df = pd.DataFrame({"P205": [0, 0, 1, 1, 0]})
        result = compute_constant_sensor_changes(df, ["P205"])
        assert result["P205_changed"].sum() == 2


class TestContradictionFeatures:
    def test_contradiction_detected(self):
        df = pd.DataFrame({
            "P101": [1, 1, 1],
            "FIT101": [0.0, 0.0, 0.0],
        })
        pairs = [{
            "actuator": "P101", "sensor": "FIT101",
            "on_value": 1, "expected": "positive",
        }]
        result = compute_contradiction_features(df, pairs)
        assert "P101_FIT101_contradiction" in result
        assert all(result["P101_FIT101_contradiction"] == 1)

    def test_no_contradiction(self):
        df = pd.DataFrame({
            "P101": [1, 1, 1],
            "FIT101": [2.5, 2.6, 2.4],
        })
        pairs = [{
            "actuator": "P101", "sensor": "FIT101",
            "on_value": 1, "expected": "positive",
        }]
        result = compute_contradiction_features(df, pairs)
        assert all(result["P101_FIT101_contradiction"] == 0)
