"""レース当日の予想生成・投稿・HTML生成・結果取得のオーケストレーション

旧 src/RacePrediction/post_daily_race.py（post_race_pred/post_pred_return/
post_daily_race_pred）を移植したもの。specifications/新設計.md で予約されている
race_day_scheduler.py に対応する。

配当結果の取得・保存（旧 src/RacePrediction/calc_returns.py の get_race_return /
save_each_race_return_csv）は src.logic.scraping.netkeiba_scraper.
scrape_race_returns_dataframe / src.managers.race_info_dataset_manager.
save_race_return_for_race_id に置き換えた。

カレンダー更新（旧 web/src/generators/date_index.py の add_race_day）は
src.managers.html_manager.add_race_day に置き換えた。
"""

import datetime
import os
import sys
from datetime import date, timedelta
from time import sleep

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import constants, paths  # noqa: E402
from src.logic.html_generator import daily_index_generator, race_page_generator  # noqa: E402
from src.logic.prediction import race_card_builder  # noqa: E402
from src.logic.scraping import netkeiba_scraper  # noqa: E402
from src.managers import (  # noqa: E402
    html_manager,
    race_card_dataset_manager,
    race_info_dataset_manager,
    race_result_dataset_manager,
)
from src.output import prediction_publisher  # noqa: E402


def post_race_pred(race_id, race_day):
    """レース予想のポスト

    Args:
        race_id(int) : race_id
        race_day(date) : レース開催日
    """
    text_path = os.path.join(paths.RACE_PREDICTION_TEXT_PATH, race_day.strftime("%Y%m%d"), f"{race_id}.txt")
    print(text_path)
    prediction_publisher.post_text_data(text_path)


def post_pred_return(place_id, race_day):
    """予想結果の配当のポスト

    Args:
        place_id(int) : place_id
        race_day(date) : レース開催日
    """
    text_path = os.path.join(
        paths.RACE_RETURN_REPORT_TEXT_PATH,
        race_day.strftime("%Y%m%d"),
        f"{constants.PLACE_LIST[place_id - 1]}_pred_score.txt",
    )
    prediction_publisher.post_text_data(text_path)


def post_daily_race_pred(race_day=date.today()):
    """一日のレースの予想をポスト

    Args:
        race_day(date) : レース開催日(初期値:今日)
    """
    date_str = race_day.strftime("%Y%m%d")
    time_id_list = race_card_dataset_manager.get_time_id_list(race_day)
    html_manager.add_race_day(race_day)
    daily_index_generator.make_daily_index_page(race_day)
    # 過去一週間のindexを再作成（リンクの生成）
    for delta_day in range(1, 8):
        past_day = race_day - timedelta(days=delta_day)
        daily_index_generator.make_daily_index_page(past_day)

    # place_id 毎に直前に処理した race_id を保持する
    last_race_by_place = {}

    while any(time_id_list):
        # レース20分前に投稿
        comp_time = datetime.datetime.now() + timedelta(minutes=20)
        str_comp_time = str(comp_time.hour).zfill(2) + str(comp_time.minute).zfill(2)
        race_time = time_id_list[0][0]
        # 実行時間を過ぎていたら投稿を実行
        if int(race_time) <= int(str_comp_time):
            race_id = time_id_list[0][1]
            try:
                # 予想の更新
                race_card_df, race_info_df = race_card_builder.make_race_card(race_id)
                # csvファイルで出力
                race_card_dataset_manager.save_race_cards(race_card_df, race_day, race_id)
                race_card_dataset_manager.save_race_info_df(race_info_df, race_day, race_id)
                race_info_dataset_manager.update_horse_name_id_map(race_card_df)
                # textの作成
                prediction_publisher.make_race_text(race_day, race_id)
                # API対策で計12レースのみ投稿
                if len(time_id_list) <= 12:
                    post_race_pred(race_id, race_day)
                    print("post:" + str(race_time + ":" + str(race_id)))
                else:
                    print("no post for API restricctinos")
                # mail送信
                prediction_publisher.send_race_pred(race_day, race_id)
            except Exception:
                print("post_error:" + str(race_time + ":" + str(race_id)))
                print(sys.exc_info())

            # htmlの作成
            place_id = int(str(race_id)[4] + str(race_id)[5])
            print("make html:" + str(race_id))
            race_page_generator.make_race_card_html(date_str, place_id, race_id)
            daily_index_generator.make_daily_index_page(race_day)

            # 直前の race_id を取得（存在すれば previous）
            previous_race_id = last_race_by_place.get(place_id)
            # 直前レースがあれば、結果の取得とhtmlを再生成(リンク更新のため)
            if previous_race_id:
                # レース結果の取得
                try:
                    results_df = netkeiba_scraper.scrape_day_race_result(previous_race_id)
                    if not results_df.empty:
                        race_result_dataset_manager.save_race_result_for_race_id(previous_race_id, results_df)
                except Exception:
                    print("Miss Make Results : ", previous_race_id)
                # 配当結果の取得
                try:
                    df_return = netkeiba_scraper.scrape_race_returns_dataframe([previous_race_id])
                    if not df_return.empty:
                        race_info_dataset_manager.save_race_return_for_race_id(previous_race_id, df_return)
                except Exception:
                    print("Miss Make Returns : ", previous_race_id)
                print("previous race html make:" + str(previous_race_id))
                race_page_generator.make_race_card_html(date_str, place_id, previous_race_id)
            # 今回処理した race_id を last_race_by_place に記録
            last_race_by_place[place_id] = race_id
            time_id_list.pop(0)

        # 1分ごとに実行
        sleep(60)

    # 最後のレースから30分待つ
    sleep(1800)
    for place_id in range(1, len(constants.PLACE_LIST) + 1):
        last_race_id = last_race_by_place.get(place_id)
        if last_race_id is not None:
            # レース結果の取得
            results_df = netkeiba_scraper.scrape_day_race_result(last_race_id)
            if not results_df.empty:
                race_result_dataset_manager.save_race_result_for_race_id(last_race_id, results_df)
            # 配当結果の取得
            df_return = netkeiba_scraper.scrape_race_returns_dataframe([last_race_id])
            if not df_return.empty:
                race_info_dataset_manager.save_race_return_for_race_id(last_race_id, df_return)

            print("previous race html make:" + str(last_race_id))
            race_page_generator.make_race_card_html(date_str, place_id, last_race_id)
