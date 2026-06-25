"""src/logic/calculators/ai_performance_calculator.py のテスト（オフライン）。

src.output.return_report の get_win_result/get_place_result/get_trio_box_result と
同じ既知データ（rank1=馬番5, rank2=馬番9, rank3=馬番7、配当は単勝160円・複勝110円・
三連複4220円）を使い、1レース判定・複数レース集計の両方を検証する。
"""

import shutil
from datetime import date, datetime

import pandas as pd
import pytest

from src.config import paths
from src.logic.calculators import ai_performance_calculator as ai
from src.managers import ai_performance_dataset_manager as m
from src.managers import race_info_dataset_manager

SAMPLE_DATE_STR = "20241020"
SAMPLE_RACE_DAY = date(2024, 10, 20)
SAMPLE_PLACE = "04_nigata"
SAMPLE_RACE_ID = "202404040601"


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    """race_card/race_returnsの出力先をtmp_path配下に切り替える。

    race_card（出馬表+score/rank）は実データ（data/race_card/20241020/202404040601.csv）
    をコピーして使う。race_returns（確定配当）は既知の値を仕込む
    （rank1=馬番5, rank2=馬番9, rank3=馬番7）。
    """
    monkeypatch.setattr(paths, "RACE_CARD_DATA_PATH", str(tmp_path / "race_card"))
    monkeypatch.setattr(race_info_dataset_manager, "RACE_RETURNS_DATA_PATH", str(tmp_path / "race_returns"))

    race_card_dir = tmp_path / "race_card" / SAMPLE_DATE_STR
    race_card_dir.mkdir(parents=True)
    shutil.copy(
        f"data/race_card/{SAMPLE_DATE_STR}/{SAMPLE_RACE_ID}.csv",
        race_card_dir / f"{SAMPLE_RACE_ID}.csv",
    )

    returns_dir = tmp_path / "race_returns" / SAMPLE_PLACE / "2024"
    returns_dir.mkdir(parents=True)
    returns_df = pd.DataFrame(
        {
            "式別": ["単勝", "複勝", "複勝", "複勝", "三連複"],
            "馬番": ["5", "5", "9", "7", "5-9-7"],
            "配当": ["160", "110", "150", "180", "4220"],
            "人気": ["1", "1", "3", "4", "12"],
        },
        index=[SAMPLE_RACE_ID] * 5,
    )
    returns_df.index.name = ""
    returns_df.to_csv(returns_dir / f"{SAMPLE_RACE_ID}.csv")

    return tmp_path


def test_calc_race_hit_returns_matches_known_result(new_roots):
    result = ai.calc_race_hit_returns(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)

    print(f"\n--- calc_race_hit_returns({SAMPLE_RACE_DAY}, {SAMPLE_RACE_ID}) ---")
    print(f"  結果: {result}")

    assert result == {
        "win": (1, 160.0),
        "place": (1, 110.0),
        "trio_box": (1, 422.0),
    }


def test_calc_race_hit_returns_returns_none_when_no_rank(new_roots, monkeypatch):
    monkeypatch.setattr(
        ai.race_card_dataset_manager, "get_race_cards", lambda race_day, race_id: pd.DataFrame({"馬番": [1, 2]})
    )

    assert ai.calc_race_hit_returns(SAMPLE_RACE_DAY, SAMPLE_RACE_ID) is None


def test_calc_race_hit_returns_returns_none_when_no_returns_data(new_roots, monkeypatch):
    monkeypatch.setattr(
        ai.race_info_dataset_manager, "get_race_return_csv_for_race", lambda race_id: pd.DataFrame()
    )

    assert ai.calc_race_hit_returns(SAMPLE_RACE_DAY, SAMPLE_RACE_ID) is None


def test_calc_race_hit_returns_returns_none_when_pick_horse_scratched(new_roots, monkeypatch):
    # rank=1（AI本命）は馬番5。本命馬が除外（出走しなかった）の場合、的中/不的中ではなく
    # 払い戻しになるため、的中率・回収率の対象から除外する（Noneを返す）。
    monkeypatch.setattr(
        ai.race_result_dataset_manager,
        "get_race_id_result",
        lambda race_id: pd.DataFrame({"馬番": [5], "着順": ["除外"]}),
    )

    assert ai.calc_race_hit_returns(SAMPLE_RACE_DAY, SAMPLE_RACE_ID) is None


def test_parse_race_id_returns_expected():
    assert ai.parse_race_id(SAMPLE_RACE_ID) == {
        "year": 2024,
        "place_id": 4,
        "times": 4,
        "days": 6,
        "race_num": 1,
    }


def test_get_current_meetings_returns_real_meetings_for_date():
    meetings = ai.get_current_meetings(date(2026, 6, 19))

    print(f"\n--- get_current_meetings(2026-06-19) ---")
    print(f"  結果: {meetings}")

    assert all(m["first_day"] <= date(2026, 6, 19) <= m["last_day"] for m in meetings)
    place_ids = [m["place_id"] for m in meetings]
    assert place_ids == sorted(place_ids)


def test_get_current_meetings_returns_empty_for_offseason_date():
    # JRAは年末年始は基本的に開催がない
    meetings = ai.get_current_meetings(date(2026, 1, 1))
    assert meetings == []


def test_current_results_weekend_end_stays_on_previous_weekend_before_wednesday():
    # 2026-06-20(土)/06-21(日)の週末を基準に考える。
    # その週末が終わった直後（月〜火）は、まだその週末を「今週」とみなさず、
    # 1つ前の週末（06-14）を指す。
    assert ai.current_results_weekend_end(date(2026, 6, 22)) == date(2026, 6, 14)  # 月
    assert ai.current_results_weekend_end(date(2026, 6, 23)) == date(2026, 6, 14)  # 火


def test_current_results_weekend_end_advances_on_wednesday():
    # 週末（06-20/06-21）から3日後の水曜日（06-24）になった時点で、
    # その週末自身（06-21）が「今週」になる。木・金（その先）も同じ週末を指す。
    # （週末自身、つまり日曜06-21時点ではまだ3日経っていないため、1つ前の週末を指す）
    assert ai.current_results_weekend_end(date(2026, 6, 21)) == date(2026, 6, 14)  # 日（週末自身）
    assert ai.current_results_weekend_end(date(2026, 6, 24)) == date(2026, 6, 21)  # 水
    assert ai.current_results_weekend_end(date(2026, 6, 26)) == date(2026, 6, 21)  # 金


def test_current_meeting_reference_day_stays_on_last_completed_weekend_through_wednesday():
    # 2026-06-22(月)が属する週の直近に終わった週末は06-20/06-21
    assert ai.current_meeting_reference_day(date(2026, 6, 22)) == date(2026, 6, 21)  # 月
    assert ai.current_meeting_reference_day(date(2026, 6, 23)) == date(2026, 6, 21)  # 火
    assert ai.current_meeting_reference_day(date(2026, 6, 24)) == date(2026, 6, 21)  # 水


def test_current_meeting_reference_day_advances_on_thursday():
    # 木曜(06-25)になった時点で、今週の週末(06-28)に切り替わる
    assert ai.current_meeting_reference_day(date(2026, 6, 25)) == date(2026, 6, 28)  # 木
    assert ai.current_meeting_reference_day(date(2026, 6, 27)) == date(2026, 6, 28)  # 土


def test_get_current_meeting_summaries_returns_place_times_and_day_number(monkeypatch):
    # current_meeting_reference_dayが06-21(日)を指すように差し替え、
    # 06-20(土)/06-21(日)それぞれの開催日目が実データと一致することを確認する
    monkeypatch.setattr(ai, "current_meeting_reference_day", lambda today: date(2026, 6, 21))

    summaries = ai.get_current_meeting_summaries(date(2026, 6, 22))

    print(f"\n--- get_current_meeting_summaries ---\n{summaries}")

    tokyo = next(s for s in summaries if s["place_id"] == 5)
    assert tokyo["days"] == [
        {"day_date": date(2026, 6, 20), "day_number": 5},
        {"day_date": date(2026, 6, 21), "day_number": 6},
    ]
    place_ids = [s["place_id"] for s in summaries]
    assert place_ids == sorted(place_ids)


def test_get_current_meeting_summaries_returns_empty_when_no_meetings(monkeypatch):
    monkeypatch.setattr(ai, "current_meeting_reference_day", lambda today: date(2026, 1, 1))

    assert ai.get_current_meeting_summaries(date(2026, 1, 1)) == []


def test_get_weekend_main_race_details_returns_winner_pick_and_hit_payout():
    # 2026-06-13(土)/06-14(日)の週末には出馬表・確定結果・配当が揃っている
    races = ai.get_weekend_main_race_details(date(2026, 6, 14))

    print(f"\n--- get_weekend_main_race_details(2026-06-14) ---")
    for r in races:
        print(f"  {r}")

    assert len(races) == 6
    assert all(r["race_day"] in (date(2026, 6, 13), date(2026, 6, 14)) for r in races)
    # 東京11R(06-13 ジューンS)はAI本命馬カネラフィーナが1着で単勝・複勝とも的中する
    tokyo_race = next(r for r in races if r["race_day"] == date(2026, 6, 13) and r["place_id"] == 5)
    assert tokyo_race["pick_name"] == tokyo_race["winner_name"] == "カネラフィーナ"
    assert tokyo_race["pick_pop"] == "3"
    assert tokyo_race["pick_finish"] == "1"
    assert tokyo_race["win_hit"] is True
    assert tokyo_race["win_payout"] == pytest.approx(510.0)
    assert tokyo_race["place_hit"] is True
    assert tokyo_race["place_payout"] == pytest.approx(210.0)
    # コース・馬場・クラス情報も確定結果側から取得して含まれる
    assert tokyo_race["race_type"] == "芝"
    assert tokyo_race["course_len"] == "1800"
    assert tokyo_race["ground_state"] == "良"
    assert tokyo_race["class"] == "オープン"


def test_get_weekend_main_race_details_returns_empty_list_when_no_schedule():
    assert ai.get_weekend_main_race_details(date(2020, 1, 5)) == []


def test_race_detail_summary_marks_pick_scratched_when_finish_is_scratch(new_roots, monkeypatch):
    # rank=1（AI本命）は馬番5。本命馬が除外（出走しなかった）の場合、pick_scratchedがTrueになる
    monkeypatch.setattr(
        ai.race_result_dataset_manager,
        "get_race_id_result",
        lambda race_id: pd.DataFrame({
            "馬番": [5, 9],
            "馬名": ["ホースA", "ホースB"],
            "着順": ["除外", "1"],
            "人気": ["1", "2"],
        }),
    )

    detail = ai._race_detail_summary(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)

    print(f"\n--- _race_detail_summary（本命馬除外）---\n{detail}")

    assert detail["pick_finish"] == "除外"
    assert detail["pick_scratched"] is True


def test_race_detail_summary_pick_scratched_false_when_finish_is_normal(new_roots):
    detail = ai._race_detail_summary(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)

    assert detail["pick_scratched"] is False


def _tokyo_2026_3rd_race_day_ids():
    """東京2026年第3回の(race_day, race_id)一覧を、永続化済みデータセットから取得する

    get_meeting_race_detailsは呼び出し側がこのペア一覧を渡す前提のため、
    テストでも実際のai_performance_dataset_manager.filter_by_meetingを使う
    （race_calendar側の「開催日目」とは日付の数え方がズレることがあるため、
    再現性のためデータセット側の実際のrace_dayを使う）。
    """
    df = m.get_ai_performance_dataset()
    meeting_df = m.filter_by_meeting(df, 2026, 5, 3)
    return [
        (datetime.strptime(row["race_day"], "%Y-%m-%d").date(), race_id)
        for race_id, row in meeting_df.iterrows()
    ]


def test_get_meeting_race_details_groups_by_day_newest_first_with_race_name():
    details = ai.get_meeting_race_details(_tokyo_2026_3rd_race_day_ids())

    print(f"\n--- get_meeting_race_details(東京2026年第3回) ---")
    for day in details:
        print(f"  {day['race_day']}: {len(day['races'])} races")

    # 開催日は新しい順
    assert [day["race_day"] for day in details] == sorted(
        (day["race_day"] for day in details), reverse=True
    )
    # データセット上の全レース数と一致する（race_calendar由来の再構築だと一部レースが
    # 取得できなくなる既知の不具合があったため、件数が欠けていないことを確認する）
    assert sum(len(day["races"]) for day in details) == len(_tokyo_2026_3rd_race_day_ids())
    # 06-13(東京11R ジューンS)が含まれ、race_id・race_nameも入っている
    day_0613 = next(day for day in details if day["race_day"] == date(2026, 6, 13))
    race = next(r for r in day_0613["races"] if r["race_id"] == "202605030311")
    assert race["race_name"] == "ジューンS"
    assert race["pick_name"] == race["winner_name"] == "カネラフィーナ"


def test_get_meeting_race_details_skips_race_when_race_result_lookup_raises(monkeypatch):
    # data/race_result/ に一部重複行が混入しているレースがあり、get_race_id_resultが
    # 例外を投げることがある（既知のデータ品質問題）。1件の異常データで開催全体の
    # 表示が止まらず、そのレースだけ読み飛ばして他のレースは正常に返ることを確認する。
    real_get_race_id_result = ai.race_result_dataset_manager.get_race_id_result

    def flaky_get_race_id_result(race_id):
        if race_id == "202605030311":
            raise KeyError("Cannot get left slice bound for non-unique label")
        return real_get_race_id_result(race_id)

    monkeypatch.setattr(ai.race_result_dataset_manager, "get_race_id_result", flaky_get_race_id_result)

    details = ai.get_meeting_race_details(_tokyo_2026_3rd_race_day_ids())

    day_0613 = next(day for day in details if day["race_day"] == date(2026, 6, 13))
    race_ids = [r["race_id"] for r in day_0613["races"]]
    assert "202605030311" not in race_ids
    assert len(race_ids) > 0


def test_list_predicted_races_returns_real_dates(new_roots):
    # new_rootsで20241020分のみ用意しているため、その1日・1レースのみ列挙される
    pairs = ai.list_predicted_races()

    print(f"\n--- list_predicted_races() ---")
    print(f"  結果: {pairs}")

    assert pairs == [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)]


def test_get_today_main_races_with_course_scrapes_course_info(monkeypatch, tmp_path):
    from src.logic.scraping import netkeiba_scraper

    today = date(2026, 6, 21)
    time_id_dir = tmp_path / "race_time_id_list"
    time_id_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "race_time": ["0950", "1545", "1530", "1520"],
            "race_id": ["202602010401", "202605030611", "202609030611", "202602010411"],
            "race_name": ["3歳未勝利", "府中牝馬S", "しらさぎS", "UHB杯"],
        }
    ).to_csv(time_id_dir / "20260621.csv", index=False)
    monkeypatch.setattr(paths, "RACE_TIME_ID_LIST_PATH", str(time_id_dir))

    def fake_scrape_race_card(race_id):
        if str(race_id) == "202605030611":
            return (["dummy"], pd.DataFrame([{"race_type": "芝", "course_len": 1800}]), pd.DataFrame())
        return (["dummy"], pd.DataFrame(), pd.DataFrame())

    monkeypatch.setattr(netkeiba_scraper, "scrape_race_card", fake_scrape_race_card)

    races = ai.get_today_main_races_with_course(today)

    print(f"\n--- get_today_main_races_with_course(2026-06-21) ---")
    print(f"  結果: {races}")

    # メインレース（11R）以外は除外され、発走時刻昇順で返る
    assert [r["race_id"] for r in races] == ["202602010411", "202609030611", "202605030611"]
    # スクレイピングに成功したレースはrace_type/course_lenが入る
    tokyo_race = next(r for r in races if r["race_id"] == "202605030611")
    assert tokyo_race["race_type"] == "芝"
    assert tokyo_race["course_len"] == 1800
    assert tokyo_race["place_id"] == 5
    # 失敗したレースはNoneのまま（呼び出し側でリンクなし表示に切り替えられる）
    hanshin_race = next(r for r in races if r["race_id"] == "202609030611")
    assert hanshin_race["race_type"] is None
    # race_dayは指定した日付がそのまま入る
    assert tokyo_race["race_day"] == today


def test_get_today_main_races_with_course_returns_empty_when_no_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_TIME_ID_LIST_PATH", str(tmp_path / "race_time_id_list"))
    assert ai.get_today_main_races_with_course(date(2026, 6, 21)) == []


def test_get_week_main_races_with_course_combines_saturday_and_sunday(monkeypatch):
    # 2026-06-22(月)は、今週(06-27/06-28)の水曜(06-24)より前なので、
    # current_schedule_weekend_endにより1つ前の週末(06-20/06-21)が対象になる
    captured_days = []

    def fake_get_today_main_races_with_course(day):
        captured_days.append(day)
        return [{"race_id": f"dummy-{day}", "race_day": day}]

    monkeypatch.setattr(ai, "get_today_main_races_with_course", fake_get_today_main_races_with_course)

    races = ai.get_week_main_races_with_course(date(2026, 6, 22))

    assert captured_days == [date(2026, 6, 20), date(2026, 6, 21)]
    assert [r["race_day"] for r in races] == [date(2026, 6, 20), date(2026, 6, 21)]


def test_current_schedule_weekend_end_stays_on_previous_weekend_before_wednesday():
    # 今週(06-27/06-28)の水曜(06-24)より前（月・火）は、まだ出馬表が公開されて
    # いないことが多いため、1つ前の週末(06-21)を指す
    assert ai.current_schedule_weekend_end(date(2026, 6, 22)) == date(2026, 6, 21)  # 月
    assert ai.current_schedule_weekend_end(date(2026, 6, 23)) == date(2026, 6, 21)  # 火


def test_current_schedule_weekend_end_advances_on_wednesday():
    # 水曜(06-24)になった時点で、今週本来の週末(06-28)に切り替わる
    assert ai.current_schedule_weekend_end(date(2026, 6, 24)) == date(2026, 6, 28)  # 水
    assert ai.current_schedule_weekend_end(date(2026, 6, 26)) == date(2026, 6, 28)  # 金
