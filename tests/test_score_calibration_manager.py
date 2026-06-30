"""src/managers/score_calibration_manager.py のテスト（オフライン）。"""

import pandas as pd
import pytest

from src.config import paths
from src.managers import score_calibration_manager as m


@pytest.fixture
def new_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCORE_CALIBRATION_PATH", str(tmp_path / "performance" / "score_calibration.csv"))
    return tmp_path


SAMPLE_CALIBRATION = pd.DataFrame(
    [
        {"band_min": 0, "band_max": 39, "n": 120, "win_rate": 2.5, "place_rate": 8.3},
        {"band_min": 40, "band_max": 49, "n": 800, "win_rate": 6.0, "place_rate": 20.0},
        {"band_min": 50, "band_max": 59, "n": 900, "win_rate": 12.0, "place_rate": 32.0},
        {"band_min": 60, "band_max": 100, "n": 400, "win_rate": 28.0, "place_rate": 55.0},
    ]
)


def test_save_and_get_score_calibration_roundtrip(new_root):
    m.save_score_calibration(SAMPLE_CALIBRATION)

    result = m.get_score_calibration()

    assert result.equals(SAMPLE_CALIBRATION)


def test_get_score_calibration_empty_when_missing(new_root):
    assert m.get_score_calibration().empty


def test_find_band_returns_matching_row(new_root):
    m.save_score_calibration(SAMPLE_CALIBRATION)

    band = m.find_band(55)

    assert band["band_min"] == 50
    assert band["band_max"] == 59
    assert band["win_rate"] == 12.0


def test_find_band_returns_none_for_none_index(new_root):
    m.save_score_calibration(SAMPLE_CALIBRATION)

    assert m.find_band(None) is None


def test_find_band_returns_none_when_no_calibration_saved(new_root):
    assert m.find_band(55) is None
