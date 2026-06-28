"""src/utils/format_data.py のテスト（オフライン）。"""

import pandas as pd

from src.utils import format_data as fd


def test_extract_entry_sub_carries_score_hitrate_when_present():
    df = pd.DataFrame({
        "馬番": ["1", "2"],
        "rank": [1, 2],
        "score": [0.8, 0.2],
        "score_hitrate": [0.123, -0.456],
    })

    result = fd.extract_entry_sub(df)

    assert result["score_hitrate"].tolist() == [0.123, -0.456]


def test_extract_entry_sub_omits_score_hitrate_when_absent():
    df = pd.DataFrame({"馬番": ["1", "2"], "rank": [1, 2], "score": [0.8, 0.2]})

    result = fd.extract_entry_sub(df)

    assert "score_hitrate" not in result.columns


def test_merge_rank_score_keeps_score_hitrate_column():
    df_race = pd.DataFrame({"馬番": [1, 2], "馬名": ["A", "B"]})
    df_analysis = pd.DataFrame({
        "馬番": ["1", "2"],
        "rank": [1, 2],
        "score": [0.8, 0.2],
        "score_hitrate": [0.123, -0.456],
    })

    result = fd.merge_rank_score(df_race, df_analysis)

    assert result["score_hitrate"].tolist() == [0.123, -0.456]
