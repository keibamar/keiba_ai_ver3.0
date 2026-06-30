"""src/logic/scheduler/race_day_scheduler.py のテスト。

post_race_pred / post_pred_return が組み立てるテキストパスと、
prediction_publisher.post_text_data への連携を確認する（オフライン）。

make_time_id_list は netkeiba.com への実通信が必要なため @pytest.mark.network を
付与する。update_weekly_time_id_list / make_html_prev_day / update_daily_html は
依存関数をmonkeypatchしてオフラインで検証する。

post_daily_race_pred はレース当日の発走時刻監視・1分間隔のポーリング・
最終レース後30分待機など sleep / datetime.now() に依存するライブループであり、
旧実装（src/RacePrediction/post_daily_race.py）も同様にユニットテスト対象外
だったため、本テストでは対象としない。
"""

import os
from datetime import date, timedelta

import pandas as pd
import pytest

from src.config import paths
from src.logic.scheduler import race_day_scheduler

FIXED_RACE_ID = "202405010101"


def test_scrape_with_retry_returns_immediately_when_not_empty(monkeypatch):
    monkeypatch.setattr(race_day_scheduler, "sleep", lambda s: None)
    calls = []

    def scrape_fn(race_id):
        calls.append(race_id)
        return pd.DataFrame({"a": [1]})

    result = race_day_scheduler._scrape_with_retry(scrape_fn, "X1")

    assert len(result) == 1
    assert calls == ["X1"]


def test_scrape_with_retry_retries_on_empty_then_succeeds(monkeypatch):
    monkeypatch.setattr(race_day_scheduler, "sleep", lambda s: None)
    calls = []

    def scrape_fn(race_id):
        calls.append(race_id)
        if len(calls) < 3:
            return pd.DataFrame()
        return pd.DataFrame({"a": [1]})

    result = race_day_scheduler._scrape_with_retry(scrape_fn, "X1", attempts=3)

    assert len(result) == 1
    assert len(calls) == 3


def test_scrape_with_retry_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(race_day_scheduler, "sleep", lambda s: None)
    calls = []

    def scrape_fn(race_id):
        calls.append(race_id)
        return pd.DataFrame()

    result = race_day_scheduler._scrape_with_retry(scrape_fn, "X1", attempts=3)

    assert result.empty
    assert len(calls) == 3


def test_update_race_card_from_result_overwrites_unconfirmed_popularity_and_weight(monkeypatch):
    # 出馬表側の人気・馬体重はライブ更新の取りこぼしで未確定("**"等)のままになることが
    # あるが、レース結果には確定人気・確定馬体重が含まれるため、結果取得時に
    # そちらで上書きできることを確認する
    race_day = date(2026, 6, 27)
    race_card_df = pd.DataFrame({
        "馬番": [1, 2, 3], "人気": ["**", "**", "**"], "馬体重(増減)": ["", "", ""],
    })
    results_df = pd.DataFrame({
        "馬番": [1, 2, 3], "人気": [2, 1, 3], "馬体重": ["472(-4)", "454(0)", "492(+2)"],
    })

    monkeypatch.setattr(
        race_day_scheduler.race_card_dataset_manager, "get_race_cards",
        lambda day, race_id: race_card_df.copy(),
    )
    saved = []
    monkeypatch.setattr(
        race_day_scheduler.race_card_dataset_manager, "save_race_cards",
        lambda df, day, race_id: saved.append(df),
    )

    race_day_scheduler._update_race_card_from_result(race_day, "X1", results_df)

    assert len(saved) == 1
    assert saved[0]["人気"].tolist() == ["2", "1", "3"]
    assert saved[0]["馬体重(増減)"].tolist() == ["472(-4)", "454(0)", "492(+2)"]


def test_update_race_card_from_result_noop_when_no_umaban_column(monkeypatch):
    saved = []
    monkeypatch.setattr(
        race_day_scheduler.race_card_dataset_manager, "save_race_cards",
        lambda df, day, race_id: saved.append(df),
    )

    race_day_scheduler._update_race_card_from_result(date(2026, 6, 27), "X1", pd.DataFrame({"着順": ["1"]}))

    assert saved == []


def test_commit_and_upload_race_day_calls_commit_and_upload_bats(monkeypatch):
    calls = []
    monkeypatch.setattr(race_day_scheduler.subprocess, "run", lambda args, **kwargs: calls.append(args[0]))

    race_day_scheduler._commit_and_upload_race_day()

    assert len(calls) == 2
    assert calls[0] == os.path.join(paths.PROJECT_ROOT, "bat", "Commit", "commit_for_race_cards.bat")
    assert calls[1] == os.path.join(paths.PROJECT_ROOT, "bat", "Deploy", "upload_to_conoha_auto.bat")


def test_commit_and_upload_race_day_swallows_errors(monkeypatch):
    def raise_error(args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(race_day_scheduler.subprocess, "run", raise_error)

    # 例外を外に伝播させない（レース当日ループを止めないため）
    race_day_scheduler._commit_and_upload_race_day()


def test_post_race_pred_posts_prediction_text(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_PREDICTION_TEXT_PATH", str(tmp_path / "race_prediction"))

    posted = []
    monkeypatch.setattr(race_day_scheduler.prediction_publisher, "post_text_data", posted.append)

    race_id = "202405010101"
    race_day = date(2024, 1, 27)

    race_day_scheduler.post_race_pred(race_id, race_day)

    expected = os.path.join(str(tmp_path / "race_prediction"), "20240127", f"{race_id}.txt")

    print(f"\n--- post_race_pred(race_id={race_id}, race_day={race_day}) ---")
    print(f"  期待パス: {expected}")
    print(f"  実際に渡されたパス: {posted}")
    print(f"  一致: {posted == [expected]}")

    assert posted == [expected]


def test_post_pred_return_posts_return_text(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_RETURN_REPORT_TEXT_PATH", str(tmp_path / "race_returns"))

    posted = []
    monkeypatch.setattr(race_day_scheduler.prediction_publisher, "post_text_data", posted.append)

    place_id = 5
    race_day = date(2024, 1, 27)

    race_day_scheduler.post_pred_return(place_id, race_day)

    expected = os.path.join(str(tmp_path / "race_returns"), "20240127", "05_tokyo_pred_score.txt")

    print(f"\n--- post_pred_return(place_id={place_id}, race_day={race_day}) ---")
    print(f"  期待パス: {expected}")
    print(f"  実際に渡されたパス: {posted}")
    print(f"  一致: {posted == [expected]}")

    assert posted == [expected]


# --- make_time_id_list / update_weekly_time_id_list -----------------------------


def test_extract_race_time_and_name_returns_expected():
    info = ["3歳未勝利", "10", "05発走", "ダ1400m", "左"]

    assert race_day_scheduler._extract_race_time(info) == "1005"
    assert race_day_scheduler._extract_race_name(info) == "3歳未勝利"


@pytest.mark.network
def test_make_time_id_list_returns_expected(monkeypatch):
    monkeypatch.setattr(
        race_day_scheduler.race_schedule_dataset_manager, "get_daily_id",
        lambda place_id, race_day: [FIXED_RACE_ID],
    )

    result = race_day_scheduler.make_time_id_list(date(2024, 1, 27))

    print(f"\n--- make_time_id_list(2024-01-27) ---")
    print(f"  結果: {result}")

    # FIXED_RACE_IDは未勝利戦（重賞ではない）のためgradeはNone
    assert result == [["1005", FIXED_RACE_ID, "3歳未勝利", None]]


def test_update_weekly_time_id_list_saves_next_7_days(monkeypatch):
    monkeypatch.setattr(race_day_scheduler, "make_time_id_list", lambda race_day: [["1000", "X", "Y"]])
    monkeypatch.setattr(race_day_scheduler.home_generator, "make_home_page", lambda: None)

    saved = []
    monkeypatch.setattr(
        race_day_scheduler.race_card_dataset_manager, "save_time_id_list",
        lambda race_day, time_id_list: saved.append((race_day, time_id_list)),
    )

    base_day = date(2024, 1, 1)
    race_day_scheduler.update_weekly_time_id_list(base_day)

    expected_days = [base_day + timedelta(days=(7 - i)) for i in range(7)]

    print(f"\n--- update_weekly_time_id_list({base_day}) ---")
    print(f"  保存された日付: {[d for d, _ in saved]}")

    assert [d for d, _ in saved] == expected_days
    assert all(time_id_list == [["1000", "X", "Y"]] for _, time_id_list in saved)


def test_update_weekly_time_id_list_skips_days_with_no_races(monkeypatch):
    monkeypatch.setattr(race_day_scheduler, "make_time_id_list", lambda race_day: [])
    monkeypatch.setattr(race_day_scheduler.home_generator, "make_home_page", lambda: None)

    saved = []
    monkeypatch.setattr(
        race_day_scheduler.race_card_dataset_manager, "save_time_id_list",
        lambda race_day, time_id_list: saved.append((race_day, time_id_list)),
    )

    race_day_scheduler.update_weekly_time_id_list(date(2024, 1, 1))

    assert saved == []


def test_update_weekly_time_id_list_resets_home_page(monkeypatch):
    monkeypatch.setattr(race_day_scheduler, "make_time_id_list", lambda race_day: [])
    monkeypatch.setattr(race_day_scheduler.race_card_dataset_manager, "save_time_id_list", lambda *a: None)

    calls = []
    monkeypatch.setattr(race_day_scheduler.home_generator, "make_home_page", lambda: calls.append(True))

    race_day_scheduler.update_weekly_time_id_list(date(2024, 1, 1))

    assert calls == [True]


# --- _upcoming_weekend_days / make_weekend_provisional_html -----------------------


def test_upcoming_weekend_days_from_thursday():
    # 2024-01-04は木曜日
    base_day = date(2024, 1, 4)

    result = race_day_scheduler._upcoming_weekend_days(base_day)

    assert result == [date(2024, 1, 6), date(2024, 1, 7)]


def test_upcoming_weekend_days_from_saturday_includes_itself():
    # 2024-01-06は土曜日
    base_day = date(2024, 1, 6)

    result = race_day_scheduler._upcoming_weekend_days(base_day)

    assert result == [date(2024, 1, 6), date(2024, 1, 7)]


def test_make_weekend_provisional_html_calls_make_html_prev_day_for_sat_and_sun(monkeypatch):
    calls = []
    monkeypatch.setattr(race_day_scheduler, "make_html_prev_day", lambda race_day: calls.append(race_day))

    # 2024-01-04は木曜日
    race_day_scheduler.make_weekend_provisional_html(date(2024, 1, 4))

    assert calls == [date(2024, 1, 6), date(2024, 1, 7)]


# --- make_html_prev_day ----------------------------------------------------------


def test_make_html_prev_day_generates_cards_and_html(monkeypatch):
    race_day = date(2026, 6, 20)
    time_id_list = [["1000", "X1"], ["1010", "X2"]]

    monkeypatch.setattr(race_day_scheduler.race_card_dataset_manager, "get_time_id_list", lambda day: time_id_list)

    calls = {"add_race_day": [], "make_daily_index_page": [], "make_race_card": [], "save_race_cards": [],
             "save_race_info_df": [], "update_horse_name_id_map": [], "make_daily_race_card_html": []}

    monkeypatch.setattr(race_day_scheduler.html_manager, "add_race_day", lambda day: calls["add_race_day"].append(day))
    monkeypatch.setattr(
        race_day_scheduler.daily_index_generator, "make_daily_index_page",
        lambda day: calls["make_daily_index_page"].append(day),
    )
    monkeypatch.setattr(
        race_day_scheduler.race_card_builder, "make_race_card",
        lambda race_id: (calls["make_race_card"].append(race_id), (pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]})))[1],
    )
    monkeypatch.setattr(
        race_day_scheduler.race_card_dataset_manager, "save_race_cards",
        lambda df, day, race_id: calls["save_race_cards"].append((day, race_id)),
    )
    monkeypatch.setattr(
        race_day_scheduler.race_card_dataset_manager, "save_race_info_df",
        lambda df, day, race_id: calls["save_race_info_df"].append((day, race_id)),
    )
    monkeypatch.setattr(
        race_day_scheduler.race_info_dataset_manager, "update_horse_name_id_map",
        lambda df: calls["update_horse_name_id_map"].append(df is not None),
    )
    monkeypatch.setattr(
        race_day_scheduler.race_page_generator, "make_daily_race_card_html",
        lambda day: calls["make_daily_race_card_html"].append(day),
    )

    race_day_scheduler.make_html_prev_day(race_day)

    print(f"\n--- make_html_prev_day({race_day}) ---")
    for key, value in calls.items():
        print(f"  {key}: {value}")

    assert calls["add_race_day"] == [race_day]
    # 当日 + 過去7日分のインデックス再生成
    assert calls["make_daily_index_page"][0] == race_day
    assert len(calls["make_daily_index_page"]) == 1 + 7 + 1  # 当日 + 過去7日 + 末尾の再生成
    assert calls["make_race_card"] == ["X1", "X2"]
    assert calls["save_race_cards"] == [(race_day, "X1"), (race_day, "X2")]
    assert calls["save_race_info_df"] == [(race_day, "X1"), (race_day, "X2")]
    assert calls["update_horse_name_id_map"] == [True, True]
    assert calls["make_daily_race_card_html"] == [race_day]


def test_make_html_prev_day_noop_when_no_time_id_list(monkeypatch):
    monkeypatch.setattr(race_day_scheduler.race_card_dataset_manager, "get_time_id_list", lambda day: [])

    add_race_day_calls = []
    monkeypatch.setattr(
        race_day_scheduler.html_manager, "add_race_day", lambda day: add_race_day_calls.append(day)
    )

    race_day_scheduler.make_html_prev_day(date(2026, 6, 20))

    assert add_race_day_calls == []


# --- update_daily_html ------------------------------------------------------------


def test_update_daily_html_saves_results_and_returns(monkeypatch):
    race_day = date(2026, 6, 20)
    time_id_list = [["1000", "X1"]]

    monkeypatch.setattr(race_day_scheduler.race_card_dataset_manager, "get_time_id_list", lambda day: time_id_list)
    monkeypatch.setattr(
        race_day_scheduler.netkeiba_scraper, "scrape_day_race_result",
        lambda race_id: pd.DataFrame({"着順": ["1"]}),
    )
    monkeypatch.setattr(
        race_day_scheduler.netkeiba_scraper, "scrape_day_race_returns",
        lambda race_id: pd.DataFrame({"式別": ["単勝"]}),
    )

    saved_results = []
    saved_returns = []
    monkeypatch.setattr(
        race_day_scheduler.race_result_dataset_manager, "save_race_result_for_race_id",
        lambda race_id, df: saved_results.append(race_id),
    )
    monkeypatch.setattr(
        race_day_scheduler.race_info_dataset_manager, "save_race_return_for_race_id",
        lambda race_id, df: saved_returns.append(race_id),
    )
    html_calls = []
    monkeypatch.setattr(
        race_day_scheduler.race_page_generator, "make_daily_race_card_html",
        lambda day: html_calls.append(day),
    )

    race_day_scheduler.update_daily_html(race_day)

    print(f"\n--- update_daily_html({race_day}) ---")
    print(f"  保存されたレース結果: {saved_results}")
    print(f"  保存された配当結果: {saved_returns}")
    print(f"  HTML再生成: {html_calls}")

    assert saved_results == ["X1"]
    assert saved_returns == ["X1"]
    assert html_calls == [race_day]
