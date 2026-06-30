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

make_time_id_list / update_weekly_time_id_list は、旧 src/RacePrediction/
make_time_id_list.py（make_time_id_list/__main__）の移植。指定日の出馬表ページを
スクレイピングしてレース名・発走時刻を集め、race_time_id_listとして保存する
（post_daily_race_pred 等が参照するデータの作成元）。

make_html_prev_day / update_daily_html は、旧 web/src/generators/
make_race_card_html.py の同名関数の移植。
"""

import datetime
import os
import re
import subprocess
import sys
from datetime import date, timedelta
from time import sleep

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import constants, paths  # noqa: E402
from src.logic.html_generator import daily_index_generator, home_generator, race_page_generator  # noqa: E402
from src.logic.prediction import race_card_builder  # noqa: E402
from src.logic.scraping import netkeiba_scraper  # noqa: E402
from src.managers import (  # noqa: E402
    html_manager,
    race_card_dataset_manager,
    race_info_dataset_manager,
    race_result_dataset_manager,
    race_schedule_dataset_manager,
)
from src.output import prediction_publisher  # noqa: E402


def _scrape_with_retry(scrape_fn, race_id, attempts=3, delay_seconds=5):
    """スクレイピング関数を、空のDataFrameが返った場合に間隔をおいて再試行する

    netkeiba側の一時的な遅延・不調（特に配当結果は確定直後にまだページが
    安定していないことがある）で空データが返ることがあるため、結果が
    空のままならこの関数内で待って再試行する。

    Args:
        scrape_fn (callable): race_idを受け取りpd.DataFrameを返すスクレイピング関数
        race_id (str): race_id
        attempts (int): 最大試行回数
        delay_seconds (int): 再試行の間隔（秒）

    Returns:
        pd.DataFrame: scrape_fnの戻り値（全試行で空だった場合は最後の空のDataFrame）
    """
    df = scrape_fn(race_id)
    for _ in range(attempts - 1):
        if not df.empty:
            break
        sleep(delay_seconds)
        df = scrape_fn(race_id)
    return df


def _update_race_card_from_result(race_day, race_id, results_df):
    """確定したレース結果の人気・馬体重を出馬表（race_card）に反映する

    出馬表の人気・馬体重は発走15〜20分前の再スクレイピングでしか更新されないため、
    そのタイミングでの取得に失敗すると未確定のままになってしまう
    （オッズ未確定時の「**」等が残る）。レース結果には確定人気・確定馬体重が
    含まれているため、結果が取得できた時点で出馬表側も上書きし、取りこぼしを
    後から回収できるようにする。

    Args:
        race_day (date): レース開催日
        race_id (str): race_id
        results_df (pd.DataFrame): scrape_day_race_resultの戻り値（人気・馬体重・馬番列を含む）
    """
    if results_df.empty or "馬番" not in results_df.columns:
        return
    race_card_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
    if race_card_df.empty or "馬番" not in race_card_df.columns:
        return

    race_card_nums = race_card_df["馬番"].astype(str)
    changed = False

    if "人気" in results_df.columns:
        pop_map = dict(zip(results_df["馬番"].astype(str), results_df["人気"].astype(str)))
        updated = race_card_nums.map(pop_map)
        race_card_df["人気"] = updated.where(updated.notna(), race_card_df.get("人気"))
        changed = True

    if "馬体重" in results_df.columns:
        weight_map = dict(zip(results_df["馬番"].astype(str), results_df["馬体重"].astype(str)))
        updated = race_card_nums.map(weight_map)
        race_card_df["馬体重(増減)"] = updated.where(updated.notna(), race_card_df.get("馬体重(増減)"))
        changed = True

    if changed:
        race_card_dataset_manager.save_race_cards(race_card_df, race_day, race_id)


def _commit_and_upload_race_day():
    """レース当日のcommit + ConoHaアップロードを行う

    post_daily_race_predのレースごとのループから呼ばれる。ConoHaへの転送は
    WinSCPのsynchronize（差分のみ転送）のため、レースごとに呼んでも負荷は小さい。
    1日の最後に1回だけアップロードする旧来の作りだと、レース中はサイトの
    予想・人気が更新されないため、レースごとに反映されるようにする。
    """
    try:
        subprocess.run(
            [os.path.join(paths.PROJECT_ROOT, "bat", "Commit", "commit_for_race_cards.bat")],
            shell=True, check=False,
        )
        subprocess.run(
            [os.path.join(paths.PROJECT_ROOT, "bat", "Deploy", "upload_to_conoha_auto.bat")],
            shell=True, check=False,
        )
    except Exception as e:
        print(f"commit/upload error: {e}")


def _extract_race_time(info):
    """scrape_race_cardのinfoから発走時刻(HHMM形式の文字列)を取り出す"""
    race_time_str = str(info[1]) + str(info[2][0]) + str(info[2][1])
    return "".join(re.findall(r"\d+", race_time_str))


def _extract_race_name(info):
    """scrape_race_cardのinfoからレース名(クラス表記)を取り出す"""
    return str(info[0])


def make_time_id_list(race_day):
    """指定日の[race_time, race_id, race_name, grade]リストを作成する

    全開催場の出馬表ページをスクレイピングし、発走時刻順にソートして返す。
    grade（G1/G2/G3。対象外のレースはNone）はnetkeiba_scraper.scrape_race_cardが
    race_info_dfに含めて返す（h1.RaceName内の重賞アイコンから判定）。

    Args:
        race_day(date) : レース開催日
    Returns:
        time_id_list(list) : [race_time, race_id, race_name, grade]のリスト(発走時刻順)
    """
    time_id_list = []
    race_id_list = race_schedule_dataset_manager.get_daily_id(0, race_day)
    for race_id in race_id_list:
        try:
            info, race_info_df, _ = netkeiba_scraper.scrape_race_card(race_id)
        except Exception:
            print(f"ℹ️ [スキップ/エラーではありません] 出馬表未公開のため取得不可: {race_id}")
            continue
        if not info:
            continue
        grade = race_info_df.iloc[0]["grade"] if not race_info_df.empty else None
        time_id_list.append([_extract_race_time(info), race_id, _extract_race_name(info), grade])

    time_id_list.sort(key=lambda row: int(row[0]))
    return time_id_list


def update_weekly_time_id_list(base_day=date.today()):
    """次の7日分のrace_time_id_listを作成・保存する（毎週木曜実行想定）

    あわせてHOMEページを再生成し、「今週のメインレース」「先週の結果」を
    新しい週の内容にリセットする。

    Args:
        base_day(date) : 基準日(初期値:今日)
    """
    for i in range(7):
        race_day = base_day + timedelta(days=(7 - i))
        time_id_list = make_time_id_list(race_day)
        if time_id_list:
            race_card_dataset_manager.save_time_id_list(race_day, time_id_list)

    home_generator.make_home_page()


def _upcoming_weekend_days(base_day):
    """base_day以降で直近の土曜・日曜の日付を返す（base_day自身が土/日ならそれを含む）

    Args:
        base_day(date) : 基準日

    Returns:
        list[date]: [直近の土曜日, 直近の日曜日]
    """
    saturday = base_day + timedelta(days=(5 - base_day.weekday()) % 7)
    sunday = saturday + timedelta(days=1)
    return [saturday, sunday]


def make_weekend_provisional_html(base_day=date.today()):
    """直近の週末（土・日）の出馬表HTMLを先行生成する（毎週木曜実行想定）

    枠順抽せん前のため、出馬表には馬名等のみが入り、AI予想（score/rank）は
    空欄になる（race_card_builder.make_race_card参照）。枠順確定後の金曜・土曜に
    make_html_prev_dayを再実行することで、AI予想入りの内容に上書きされる。
    update_weekly_time_id_list で保存済みのrace_time_id_listが前提となる。

    Args:
        base_day(date) : 基準日(初期値:今日)
    """
    for race_day in _upcoming_weekend_days(base_day):
        make_html_prev_day(race_day)


def make_html_prev_day(race_day):
    """翌日分の出馬表・予想・HTMLを事前生成する（毎週金・土実行想定）

    race_dayにはあらかじめ update_weekly_time_id_list で保存された
    race_time_id_listが存在している必要がある。

    Args:
        race_day(date) : レース開催日(通常は実行日の翌日)
    """
    time_id_list = race_card_dataset_manager.get_time_id_list(race_day)
    if not time_id_list:
        return

    html_manager.add_race_day(race_day)
    daily_index_generator.make_daily_index_page(race_day)
    # 過去一週間のindexを再作成（リンクの生成）
    for delta_day in range(1, 8):
        past_day = race_day - timedelta(days=delta_day)
        daily_index_generator.make_daily_index_page(past_day)

    for _, race_id in time_id_list:
        try:
            race_card_df, race_info_df = race_card_builder.make_race_card(race_id)
            race_card_dataset_manager.save_race_cards(race_card_df, race_day, race_id)
            race_card_dataset_manager.save_race_info_df(race_info_df, race_day, race_id)
            race_info_dataset_manager.update_horse_name_id_map(race_card_df)
        except Exception:
            print(f"Make RaceCard Error: {race_id}")

    race_page_generator.make_daily_race_card_html(race_day)
    daily_index_generator.make_daily_index_page(race_day)


def update_daily_html(race_day=date.today()):
    """当日分のレース結果・配当結果を取得し、HTMLを再生成する（毎週土・日実行想定）

    Args:
        race_day(date) : レース開催日(初期値:今日)
    """
    time_id_list = race_card_dataset_manager.get_time_id_list(race_day)
    for _, race_id in time_id_list:
        try:
            results_df = _scrape_with_retry(netkeiba_scraper.scrape_day_race_result, race_id)
            if not results_df.empty:
                race_result_dataset_manager.save_race_result_for_race_id(race_id, results_df)
                _update_race_card_from_result(race_day, race_id, results_df)
        except Exception:
            print("Miss Make Results : ", race_id)
        try:
            # db.netkeiba.com（scrape_race_returns_dataframe）は当該シーズン中の
            # レースの配当結果がまだ反映されておらず空になるため、結果と同じ
            # 速報ページから取得するscrape_day_race_returnsを使う。netkeiba側の
            # 一時的な不調で空が返ることがあるため再試行つきで取得する
            df_return = _scrape_with_retry(netkeiba_scraper.scrape_day_race_returns, race_id)
            if not df_return.empty:
                race_info_dataset_manager.save_race_return_for_race_id(race_id, df_return)
        except Exception:
            print("Miss Make Returns : ", race_id)

    race_page_generator.make_daily_race_card_html(race_day)


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
                    results_df = _scrape_with_retry(netkeiba_scraper.scrape_day_race_result, previous_race_id)
                    if not results_df.empty:
                        race_result_dataset_manager.save_race_result_for_race_id(previous_race_id, results_df)
                        _update_race_card_from_result(race_day, previous_race_id, results_df)
                except Exception:
                    print("Miss Make Results : ", previous_race_id)
                # 配当結果の取得（db.netkeiba.comは当該シーズン中のレースが未反映のため、
                # 結果と同じ速報ページから取得するscrape_day_race_returnsを使う。netkeiba側の
                # 一時的な不調で空が返ることがあるため再試行つきで取得する）
                try:
                    df_return = _scrape_with_retry(netkeiba_scraper.scrape_day_race_returns, previous_race_id)
                    if not df_return.empty:
                        race_info_dataset_manager.save_race_return_for_race_id(previous_race_id, df_return)
                except Exception:
                    print("Miss Make Returns : ", previous_race_id)
                print("previous race html make:" + str(previous_race_id))
                race_page_generator.make_race_card_html(date_str, place_id, previous_race_id)
            # 今回処理した race_id を last_race_by_place に記録
            last_race_by_place[place_id] = race_id
            time_id_list.pop(0)

            # レースごとにcommit+ConoHaアップロードし、当日中の予想・人気更新をサイトに反映する
            _commit_and_upload_race_day()

        # 1分ごとに実行
        sleep(60)

    # 最後のレースから30分待つ
    sleep(1800)
    for place_id in range(1, len(constants.PLACE_LIST) + 1):
        last_race_id = last_race_by_place.get(place_id)
        if last_race_id is not None:
            # レース結果の取得
            results_df = _scrape_with_retry(netkeiba_scraper.scrape_day_race_result, last_race_id)
            if not results_df.empty:
                race_result_dataset_manager.save_race_result_for_race_id(last_race_id, results_df)
                _update_race_card_from_result(race_day, last_race_id, results_df)
            # 配当結果の取得（db.netkeiba.comは当該シーズン中のレースが未反映のため、
            # 結果と同じ速報ページから取得するscrape_day_race_returnsを使う。netkeiba側の
            # 一時的な不調で空が返ることがあるため再試行つきで取得する）
            df_return = _scrape_with_retry(netkeiba_scraper.scrape_day_race_returns, last_race_id)
            if not df_return.empty:
                race_info_dataset_manager.save_race_return_for_race_id(last_race_id, df_return)

            print("previous race html make:" + str(last_race_id))
            race_page_generator.make_race_card_html(date_str, place_id, last_race_id)
