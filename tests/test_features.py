import pandas as pd
import pytest

from src.features import build_features, load_feature_set

INDEX = pd.date_range("2024-03-04", periods=3, freq="D", tz="UTC")
BARS = pd.DataFrame(
    {
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.0, 102.0, 101.0],
        "volume": [10.0, 20.0, 30.0],
    },
    index=INDEX,
)


def test_feature_set_declares_its_columns_and_definitions():
    feature_set = load_feature_set("momentum_volatility_volume")
    assert feature_set.columns == ("r1", "r5", "r20", "dist_ma", "vol", "vol_ratio")
    assert set(feature_set.descriptions) == set(feature_set.columns)
    assert "position" not in feature_set.columns


def test_void_returns_only_timestamps():
    frame = build_features({"SPY": BARS}, "void")
    rows = frame.xs("SPY", level="symbol")
    assert list(rows.columns) == []
    assert rows.index.equals(INDEX)


def test_columns_come_from_the_declaration():
    frame = build_features(
        {"SPY": BARS},
        "momentum_volatility_volume",
        ma_window=2,
        vol_window=2,
        volume_window=2,
    )
    rows = frame.xs("SPY", level="symbol")
    assert list(rows.columns) == ["r1", "r5", "r20", "dist_ma", "vol", "vol_ratio"]
    assert rows.index.equals(INDEX)
    assert rows.loc[INDEX[1], "r1"] == pytest.approx(0.02)
