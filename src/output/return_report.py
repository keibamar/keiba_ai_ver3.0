"""配当結果レポート（回収率テキスト）の生成・X(Twitter)配信を担うモジュール

旧 src/RacePrediction/calc_returns.py（get_win_result/get_place_result/
get_trio_box_result/get_each_race_retutn_csv/post_race_rerurns）、
src/RacePrediction/make_text.py（write_*_hit_text/make_return_text）からの移植。

旧実装の get_trio_box_result は「式別」列を "3連複"（算用数字）でフィルタしていたが、
race_info_dataset_manager.get_race_return_csv_for_race が返す「式別」は
"三連複"（漢数字）表記のため、新実装ではこちらでフィルタする
（src/datasets/race_info/transform.py の format_type_returns_dataframe を参照）。

calc_returns.py に残る孤立した回収率レポートCSV機能
（get_quinella_*/calc_day_race_return*/save_*/run_all_year/get_race_return等）は対象外。
"""

import math
import os
import re
from datetime import date

from src.config import paths
from src.config.constants import NAME_LIST, PLACE_LIST
from src.managers import race_card_dataset_manager, race_info_dataset_manager, race_schedule_dataset_manager
from src.output import prediction_publisher


def return_report_error(e):
    """エラー時動作を記載する

    Args:
        e (Exception) : エラー内容
    """
    print(f"❌ [要確認/エラー] {__name__}:{__file__}")
    print(f"❌ [要確認/エラー] {e.__class__.__name__}: {e}")


def get_win_result(race_day, race_id_list):
    """指数一位馬の単勝回収率・的中率・的中レースを出力

    Args:
        race_day(date) : レース開催日
        race_id_list(list) : 回収率を計算するrace_idのリスト
    Returns:
        win_hit_rate(int) : 的中率
        win_return_rate(int) : 回収率
        win_hit_race(str) : 的中レース
    """
    win_hit_rate = 0
    win_return_rate = 0
    win_hit_race = ""
    race_num_diff = 0
    for race_id in race_id_list:
        # 予想結果の取得
        pred_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
        if "rank" not in pred_df.columns:
            print(f"ℹ️ [スキップ/エラーではありません] 予想スコア未生成（rank列なし）: {race_id}")
            race_num_diff += 1
            continue

        # 配当データの取得
        returns_df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
        if returns_df.empty:
            race_num_diff += 1
            continue

        # 1着的中率・回収率
        win_df = returns_df[returns_df["式別"] == "単勝"].reset_index(drop=True)
        for i in range(len(win_df)):
            num = int(win_df.at[i, "馬番"])
            if pred_df.at[num - 1, "rank"] == 1:
                win_hit_rate += 1
                win_return_rate = win_return_rate + float(win_df.at[i, "配当"])
                win_hit_race += str(int(str(race_id)[10] + str(race_id)[11])) + " "

    if not len(race_id_list) == race_num_diff:
        win_hit_rate = win_hit_rate / (len(race_id_list) - race_num_diff)
        win_return_rate = win_return_rate / (len(race_id_list) - race_num_diff)
    return win_hit_rate, win_return_rate, win_hit_race


def get_place_result(race_day, race_id_list):
    """指数一位馬の複勝回収率・的中率・的中レースを出力

    Args:
        race_day(date) : レース開催日
        race_id_list(list) : 回収率を計算するrace_idのリスト
    Returns:
        place_hit_rate(int) : 的中率
        place_return_rate(int) : 回収率
        place_hit_race(str) : 的中レース
    """
    place_hit_rate = 0
    place_return_rate = 0
    place_hit_race = ""
    race_num_diff = 0
    for race_id in race_id_list:
        # 予想結果の取得
        pred_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
        if "rank" not in pred_df.columns:
            print(f"ℹ️ [スキップ/エラーではありません] 予想スコア未生成（rank列なし）: {race_id}")
            race_num_diff += 1
            continue

        # 配当データの取得
        returns_df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
        if returns_df.empty:
            race_num_diff += 1
            continue

        # ３着内率・複勝回収率
        place_df = returns_df[returns_df["式別"] == "複勝"].reset_index(drop=True)
        for i in range(len(place_df)):
            num = int(place_df.at[i, "馬番"])
            if pred_df.at[num - 1, "rank"] == 1:
                place_hit_rate = place_hit_rate + 1
                if type(place_df.at[i, "配当"]) == str:
                    place_df.at[i, "配当"] = re.sub(r"\D", "", place_df.at[i, "配当"])
                place_return_rate = place_return_rate + float(place_df.at[i, "配当"])
                place_hit_race += str(int(str(race_id)[10] + str(race_id)[11])) + " "

    if not len(race_id_list) == race_num_diff:
        place_hit_rate = place_hit_rate / (len(race_id_list) - race_num_diff)
        place_return_rate = place_return_rate / (len(race_id_list) - race_num_diff)
    return place_hit_rate, place_return_rate, place_hit_race


def get_trio_box_result(race_day, race_id_list, box_num):
    """三連複BOXの回収率・的中率・的中レースを出力

    Args:
        race_day(date) : レース開催日
        race_id_list(list) : 回収率を計算するrace_idのリスト
        box_num(int) : ボックスの頭数
    Returns:
        trio_box_hit_rate(int) : 的中率
        trio_box_return_rate(int) : 回収率
        trio_box_hit_race(str) : 的中レース
    """
    trio_box_hit_rate = 0
    trio_box_return_rate = 0
    trio_box_hit_race = ""
    race_num_diff = 0
    for race_id in race_id_list:
        # 予想結果の取得
        pred_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
        if "rank" not in pred_df.columns:
            print(f"ℹ️ [スキップ/エラーではありません] 予想スコア未生成（rank列なし）: {race_id}")
            race_num_diff += 1
            continue

        # 配当データの取得
        returns_df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
        if returns_df.empty:
            race_num_diff += 1
            continue

        # 三連複BOX的中率・回収率
        trio_df = returns_df[returns_df["式別"] == "三連複"].reset_index(drop=True)
        for i in range(len(trio_df)):
            # 三連複の馬番の取得
            num_list = re.findall(r"\d+", trio_df.at[i, "馬番"])
            # １着馬・２着馬・３着馬の予想順位を取得
            rank_1 = pred_df[pred_df["馬番"] == int(num_list[0])].reset_index(drop=True).at[0, "rank"]
            rank_2 = pred_df[pred_df["馬番"] == int(num_list[1])].reset_index(drop=True).at[0, "rank"]
            rank_3 = pred_df[pred_df["馬番"] == int(num_list[2])].reset_index(drop=True).at[0, "rank"]

            # ランキング５位以内で３頭の場合
            eval_1 = rank_1 <= box_num
            eval_2 = rank_2 <= box_num
            eval_3 = rank_3 <= box_num
            if eval_1 and eval_2 and eval_3:
                trio_box_hit_rate += 1
                if type(trio_df.at[i, "配当"]) == str:
                    trio_df.at[i, "配当"] = re.sub(r"\D", "", trio_df.at[i, "配当"])
                trio_box_return_rate += float(trio_df.at[i, "配当"])
                trio_box_hit_race += str(int(str(race_id)[10] + str(race_id)[11])) + " "

    if not len(race_id_list) == race_num_diff:
        trio_box_hit_rate = trio_box_hit_rate / (len(race_id_list) - race_num_diff)
        trio_box_return_rate = (trio_box_return_rate / (len(race_id_list) - race_num_diff)) / math.comb(box_num, 3)
    return trio_box_hit_rate, trio_box_return_rate, trio_box_hit_race


def write_win_hit_text(race_day, race_id_list, text_file):
    """単勝の予想結果をテキストに記述

    Args:
        race_day(date) : レース日
        race_id_list(list) : 結果を計算するレースidのリスト
        text_file(file) : テキストファイル
    """
    # 単勝回収率・的中率・的中レースを抽出
    _, win_return_rate, win_hit_race = get_win_result(race_day, race_id_list)
    text_file.write("◎単勝回収率:" + str(round(win_return_rate, 1)) + "%  " + "(的中レース:" + win_hit_race + "R)\n")


def write_place_hit_text(race_day, race_id_list, text_file):
    """複勝の予想結果をテキストに記述

    Args:
        race_day(date) : レース日
        race_id_list(list) : 結果を計算するレースidのリスト
        text_file(file) : テキストファイル
    """
    # 複勝回収率・的中率・的中レースを抽出
    _, place_return_rate, place_hit_race = get_place_result(race_day, race_id_list)
    text_file.write("◎複勝回収率:" + str(round(place_return_rate, 1)) + "%  " + "(的中レース:" + place_hit_race + "R)\n")


def write_trio_hit_text(race_day, race_id_list, text_file):
    """三連複の予想結果をテキストに記述

    Args:
        race_day(date) : レース日
        race_id_list(list) : 結果を計算するレースidのリスト
        text_file(file) : テキストファイル
    """
    # 上位5頭三連複BOX回収率・的中率・的中レースを抽出
    _, trio5_return_rate, trio5_hit_race = get_trio_box_result(race_day, race_id_list, box_num=5)
    text_file.write("三連複(5頭BOX)回収率:" + str(round(trio5_return_rate, 1)) + "%  " + "(的中レース:" + trio5_hit_race + "R)\n")


def make_return_text(place_id, race_day=date.today()):
    """当日の予想結果のテキスト作成

    Args:
        place_id(int) : place_id
        race_day(date) : レース開催日
    """
    try:
        # 今日のid_listを取得
        race_id_list = race_schedule_dataset_manager.get_daily_id(place_id, race_day)

        # 的中率・回収率の計算
        if race_id_list:
            # テキストファイルの準備
            folder_path = os.path.join(paths.RACE_RETURN_REPORT_TEXT_PATH, race_day.strftime("%Y%m%d"))
            os.makedirs(folder_path, exist_ok=True)
            text_data_path = os.path.join(folder_path, f"{PLACE_LIST[place_id - 1]}_pred_score.txt")
            f = open(text_data_path, "w", encoding="UTF-8")

            # 日付の出力
            f.write(str(race_day.year) + "/" + str(race_day.month) + "/" + str(race_day.day) + " ")
            # 開催競馬場の出力
            f.write(NAME_LIST[place_id - 1] + "競馬場\n\n" + "MAR競馬予想 回収率\n")

            # 予想結果の出力
            write_win_hit_text(race_day, race_id_list, f)
            write_place_hit_text(race_day, race_id_list, f)
            write_trio_hit_text(race_day, race_id_list, f)
            f.write("\n\n")

            # タグの出力
            f.write("#MAR競馬予想\n")
            f.write("#競馬予想AI\n")
            f.write("#競馬 #競馬予想\n")
            f.write("#" + NAME_LIST[place_id - 1] + "競馬場\n")
            f.close()
    except Exception as e:
        return_report_error(e)


def post_race_returns(place_id, race_day):
    """配当結果レポートをポスト

    Args:
        place_id(int) : place_id
        race_day(date) : レース開催日
    """
    text_data_path = os.path.join(
        paths.RACE_RETURN_REPORT_TEXT_PATH, race_day.strftime("%Y%m%d"), f"{PLACE_LIST[place_id - 1]}_pred_score.txt"
    )
    prediction_publisher.post_text_data(text_data_path)


def post_daily_race_returns(race_day=date.today()):
    """指定日に開催のあった全place_idについて、回収率レポートを生成・投稿する

    旧 src/RacePrediction/calc_returns.py の post_race_rerurns（当日全場分の
    呼び出し）の移植。開催のなかったplace_idはスキップする。

    Args:
        race_day(date) : レース開催日(初期値:今日)
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        race_id_list = race_schedule_dataset_manager.get_daily_id(place_id, race_day)
        if not race_id_list:
            continue
        make_return_text(place_id, race_day)
        post_race_returns(place_id, race_day)
