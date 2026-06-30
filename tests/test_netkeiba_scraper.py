"""src/logic/scraping/netkeiba_scraper.py のテスト

parse_race_info_tokens は純粋関数なのでオフラインで検証する。
scrape_race_results / scrape_day_race_result / scrape_race_returns_dataframe /
scrape_race_card はnetkeiba.comへの実通信が必要なため @pytest.mark.network を付与し、
確定済みの固定race_id（FIXED_RACE_ID）について、既知の期待値との比較で検証する。

scrape_race_returns_dataframeについては、旧 src/legacy_datasets/race_returns.py の
同名関数はformat_type_returns_dataframeの戻り値に対する.set_index(0)がKeyErrorとなり
処理が失敗するため（詳細はnetkeiba_scraper.scrape_race_returns_dataframeのdocstring参照）、
新旧比較ではなく、data/RaceReturns/05_tokyo/2024_race_returns.csv に保存済みの結果から
判明している期待値との比較で検証する。
"""

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.datasets.race_info import model as race_info_model
from src.datasets.race_result import transform
from src.logic.scraping import common, netkeiba_scraper

# 2024年1月27日 東京1回1日目1R（確定済みのレース結果ページ）
FIXED_RACE_ID = "202405010101"


def test_parse_race_info_tokens():
    tokens = ["3歳未勝利", "ダート", "1400m", "晴", "良", "2024年01月27日", "10時05分"]

    info = transform.parse_race_info_tokens(tokens)

    assert info["race_type"] == "ダート"
    assert info["course_len"] == 1400
    assert info["class"] == "未勝利"
    assert info["weather"] == "晴"
    assert info["ground_state"] == "良"
    assert info["date"] == "2024年01月27日"


def test_parse_race_info_tokens_handles_turf_and_obstacle():
    assert transform.parse_race_info_tokens(["芝", "2000m", "3歳以上オープン"])["race_type"] == "芝"
    assert transform.parse_race_info_tokens(["障害3000m", "障害3歳以上オープン"])["race_type"] == "障害"


@pytest.mark.network
def test_scrape_race_results_returns_expected():
    df = netkeiba_scraper.scrape_race_results(FIXED_RACE_ID)

    print(f"\n--- scrape_race_results({FIXED_RACE_ID}) ---")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns.tolist()}")
    print(df.to_string())

    assert df.shape == (16, 23)
    assert df.columns.tolist() == [
        "着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差", "通過", "上り",
        "単勝", "人気", "馬体重", "調教師", "course_len", "weather", "race_type", "ground_state",
        "date", "class", "horse_id", "jockey_id",
    ]
    assert df.index.unique().tolist() == [FIXED_RACE_ID]

    first = df.iloc[0]
    assert first[["着順", "馬番", "馬名", "タイム", "単勝", "course_len", "race_type", "horse_id", "jockey_id"]].tolist() == [
        "1", "7", "エースアビリティ", "0:1:26.9", "1.4", 1400, "ダート", "2021102098", "05339",
    ]


@pytest.mark.network
def test_scrape_day_race_result_returns_expected():
    df = netkeiba_scraper.scrape_day_race_result(FIXED_RACE_ID)

    print(f"\n--- scrape_day_race_result({FIXED_RACE_ID}) ---")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns.tolist()}")
    print(df.to_string())

    assert df.shape == (16, 15)
    assert df.columns.tolist() == [
        "着順", "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差", "人気", "単勝", "上り", "通過", "厩舎", "馬体重",
    ]
    assert df.index.unique().tolist() == [FIXED_RACE_ID]

    first = df.iloc[0]
    assert first[["着順", "馬番", "馬名", "タイム", "単勝", "厩舎", "馬体重"]].tolist() == [
        1, 7, "エースアビリティ", "0:1:26.9", 1.4, "美浦 堀内", "480(+2)",
    ]


@pytest.mark.network
def test_scrape_race_returns_dataframe_matches_known_result():
    # data/RaceReturns/05_tokyo/2024_race_returns.csv に保存済みの確定結果から判明している期待値
    expected = pd.DataFrame(
        [
            ["単勝", "7", "140", "1"],
            ["複勝", "7", "110", "1"],
            ["複勝", "6", "230", "3"],
            ["複勝", "13", "400", "6"],
            ["枠連", "3-4", "790", "2"],
            ["馬連", "6-7", "760", "2"],
            ["ワイド", "6-7", "350", "2"],
            ["ワイド", "7-13", "620", "7"],
            ["ワイド", "6-13", "2310", "17"],
            ["馬単", "7→6", "1130", "4"],
            ["三連複", "6-7-13", "4240", "11"],
            ["三連単", "7→6→13", "10490", "30"],
        ],
        columns=race_info_model.RACE_RETURNS_COLUMNS,
        index=[FIXED_RACE_ID] * 12,
    )

    new_result = netkeiba_scraper.scrape_race_returns_dataframe([FIXED_RACE_ID])

    print(f"\n--- scrape_race_returns_dataframe([{FIXED_RACE_ID}]) ---")
    print(f"shape: {new_result.shape}")
    print(new_result.to_string())

    assert new_result.reset_index(drop=True).equals(expected.reset_index(drop=True))
    assert new_result.index.tolist() == expected.index.tolist()


# db.netkeiba.com（scrape_race_returns_dataframe）は当該シーズン中のレースの配当結果が
# まだ反映されておらず空になるため、当日〜近日中のレースを取得する場合は
# race.netkeiba.comの速報ページ（result.html）内のPayout_Detail_Tableから取得する
# scrape_day_race_returnsを使う。2026/6/28 函館1Rの確定済み配当結果で検証する。
RECENT_FIXED_RACE_ID = "202602010601"


def test_scrape_day_race_returns_strips_comma_from_large_ninki(monkeypatch):
    # 三連単等は組み合わせ数が多く、人気が"1,579人気"のようにカンマ区切りに
    # なることがある。カンマを取り除かずint変換すると、HTML生成側
    # （race_page_generator.generate_payout_table_html）でクラッシュする不具合があった
    html = """
    <table class="Payout_Detail_Table">
      <tr class="Tan3">
        <th>3連単</th>
        <td class="Result"><ul><li><span>7</span></li><li><span>1</span></li><li><span>6</span></li></ul></td>
        <td class="Payout"><span>999,999円</span></td>
        <td class="Ninki"><span>1,579人気</span></td>
      </tr>
    </table>
    """
    monkeypatch.setattr(common, "url_exists", lambda url: True)
    monkeypatch.setattr(common, "fetch_soup", lambda url: BeautifulSoup(html, "html.parser"))
    monkeypatch.setattr(common, "validate_soup", lambda *a, **k: True)

    result = netkeiba_scraper.scrape_day_race_returns("999999999999")

    assert result.loc["999999999999", "人気"] == "1579"
    assert result.loc["999999999999", "配当"] == "999999"


@pytest.mark.parametrize(
    "grade_classes, expected",
    [
        (["Icon_GradeType", "Icon_GradeType1"], "G1"),
        (["Icon_GradeType", "Icon_GradeType2"], "G2"),
        (["Icon_GradeType", "Icon_GradeType3"], "G3"),
        # Icon_GradeType13は重賞の有無を問わず付く「国際」マークのため対象外
        (["Icon_GradeType", "Icon_GradeType13", "Icon_GradePos01"], None),
        # Listed(15)・OP特別(17)等、G1/G2/G3以外は対象外
        (["Icon_GradeType", "Icon_GradeType15"], None),
        (["Icon_GradeType", "Icon_GradeType17", "Icon_GradePos01"], None),
    ],
)
def test_extract_race_grade_maps_icon_class_to_grade(grade_classes, expected):
    class_attr = " ".join(grade_classes)
    soup = BeautifulSoup(f'<h1 class="RaceName">レース名<span class="{class_attr}"></span></h1>', "html.parser")

    assert netkeiba_scraper._extract_race_grade(soup) == expected


def test_extract_race_grade_returns_none_when_no_grade_icon():
    soup = BeautifulSoup('<h1 class="RaceName">3歳未勝利</h1>', "html.parser")

    assert netkeiba_scraper._extract_race_grade(soup) is None


def test_extract_race_grade_returns_g1_when_international_icon_also_present():
    # 函館記念のように複数のIcon_GradeType系spanが並ぶ場合、国際マーク（13）より先に
    # 重賞アイコンが見つかってもどちらの順でも正しくG1/G2/G3を判定できる
    soup = BeautifulSoup(
        '<h1 class="RaceName">安田記念'
        '<span class="Icon_GradeType Icon_GradeType1"></span>'
        '<span class="Icon_GradeType Icon_GradeType13 Icon_GradePos01"></span>'
        "</h1>",
        "html.parser",
    )

    assert netkeiba_scraper._extract_race_grade(soup) == "G1"


@pytest.mark.network
def test_scrape_race_card_extracts_grade_for_graded_race():
    # 函館記念（G3）
    _, race_info_df, _ = netkeiba_scraper.scrape_race_card("202602010611")

    assert race_info_df.iloc[0]["grade"] == "G3"


@pytest.mark.network
def test_scrape_day_race_returns_matches_known_result():
    expected = pd.DataFrame(
        [
            ["単勝", "7", "240", "1"],
            ["複勝", "7", "120", "1"],
            ["複勝", "1", "210", "5"],
            ["複勝", "6", "1790", "8"],
            ["馬連", "1-7", "1150", "7"],
            ["ワイド", "1-7", "460", "8"],
            ["ワイド", "6-7", "4850", "23"],
            ["ワイド", "1-6", "4120", "22"],
            ["馬単", "7→1", "1840", "9"],
            ["三連複", "1-6-7", "19590", "37"],
            ["三連単", "7→1→6", "61470", "122"],
        ],
        columns=race_info_model.RACE_RETURNS_COLUMNS,
        index=[RECENT_FIXED_RACE_ID] * 11,
    )

    result = netkeiba_scraper.scrape_day_race_returns(RECENT_FIXED_RACE_ID)

    print(f"\n--- scrape_day_race_returns({RECENT_FIXED_RACE_ID}) ---")
    print(f"shape: {result.shape}")
    print(result.to_string())

    assert result.reset_index(drop=True).equals(expected.reset_index(drop=True))
    assert result.index.tolist() == expected.index.tolist()


@pytest.mark.network
def test_scrape_race_card_returns_expected():
    info, race_info_df, race_card_df = netkeiba_scraper.scrape_race_card(FIXED_RACE_ID)

    print(f"\n--- scrape_race_card({FIXED_RACE_ID}) ---")
    print(f"info: {info}")
    print(f"race_info_df:\n{race_info_df.to_string()}")
    print(f"race_card_df (shape={race_card_df.shape}):\n{race_card_df.to_string()}")

    assert info == [
        "3歳未勝利", "10", "05発走", "ダ1400m", "左", "天候", "晴", "馬場", "良",
        "サラ系３歳", "未勝利", "混", "指", "馬齢", "16頭",
    ]
    assert race_info_df.to_dict("records") == [
        {
            "race_type": "ダート", "course_len": 1400, "weather": "晴", "ground_state": "良", "class": "未勝利",
            "grade": None,
        }
    ]

    assert race_card_df.shape == (16, 13)
    assert race_card_df.columns.tolist() == [
        "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "厩舎", "馬体重(増減)", "オッズ", "人気",
        "所属", "horse_id", "jockey_id",
    ]
    assert race_card_df.index.unique().tolist() == [FIXED_RACE_ID]
    # この race_id は既に終了したレースのため、shutuba.html上の現在のオッズ・人気は
    # 確定値ではなく未確定/失効を示すプレースホルダー（---.- / **）になる。
    # 開催当日（発走前）にスクレイピングした場合は、ここに実際のオッズ・人気が入る。
    assert race_card_df.iloc[0].tolist() == [
        1, 1, "アフロマン", "牡3", 57.0, "木幡育", "萱野", "410(+16)", "---.-", "**", "美浦", "2021107090", "01167",
    ]
