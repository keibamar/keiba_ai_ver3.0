import os
import sys

from datetime import date, timedelta
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import name_header
import get_race_id

import calc_returns
import race_card

def make_test_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def extract_top5_pred(race_data_df):
    """予想結果の上位5頭のリストを返す
        Args:
            race_data_df(pd.DataFrame) : 出馬表データセット
        Returns:
            result_list(pd.DataFrame) : 上位5頭のデータセット(昇順)
    """
    result_list = []
    for i in range(1, 6):
        temp = race_data_df[race_data_df["rank"] == i].reset_index(drop = True)
        if not temp.empty:
            num = temp.at[0,"馬番"]
            name = temp.at[0,"馬名"]
            result_list.append([num, name])
    return result_list

def write_win_hit_text(race_day, race_id_list, text_file):
    """単勝の予想結果をテキストに記述
        Args:
            race_day(date) : レース日
            race_id_list(list) : 結果を計算するレースidのリスト
            text_file(file) : テキストファイル
    """
    # 単勝回収率・的中率・的中レースを抽出
    win_hit_rate, win_return_rate, win_hit_race = calc_returns.get_win_result(race_day, race_id_list)
    text_file.write("◎単勝回収率:" + str(round(win_return_rate, 1)) + "%  " + "(的中レース:" + win_hit_race + "R)\n")

def write_place_hit_text(race_day, race_id_list, text_file):
    """複勝の予想結果をテキストに記述
        Args:
            race_day(date) : レース日
            race_id_list(list) : 結果を計算するレースidのリスト
            text_file(file) : テキストファイル
    """
     # 複勝回収率・的中率・敵流レースを抽出
    place_hit_rate, place_return_rate, place_hit_race = calc_returns.get_place_result(race_day, race_id_list)
    text_file.write("◎複勝回収率:" + str(round(place_return_rate, 1)) + "%  " + "(的中レース:" + place_hit_race + "R)\n")

def write_trio_hit_text(race_day, race_id_list, text_file):
    """三連複の予想結果をテキストに記述
        Args:
            race_day(date) : レース日
            race_id_list(list) : 結果を計算するレースidのリスト
            text_file(file) : テキストファイル
    """
    # 上位5頭三連複BOX回収率・的中率・敵流レースを抽出
    trio5_hit_rate, trio5_return_rate, trio5_hit_race = calc_returns.get_trio_box_result(race_day, race_id_list, box_num = 5)
    text_file.write("三連複(5頭BOX)回収率:" + str(round(trio5_return_rate, 1)) + "%  " + "(的中レース:" + trio5_hit_race + "R)\n")

def make_race_text(race_day, race_id):
    """レースの予想のテキスト作成
        Args:
            race_day(date) : レース開催日
            race_id(int) : race_id
    """
    # 予想結果を抽出
    race_data_df = race_card.get_race_cards(race_day, race_id)
    if not "rank" in race_data_df.columns:
        print("not rank:" + str(race_id))
        return
    try:
        # 予想結果から上位5頭を抽出
        pred_list = extract_top5_pred(race_data_df)

        # テキストファイルの準備
        folder_path = name_header.TEXT_PATH + "race_prediction/" + race_day.strftime("%Y%m%d")
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        text_data_path = folder_path + "//" + str(race_id) + ".txt"
        
        f = open(text_data_path, "w", encoding = "UTF-8")

        # 開催情報の抽出
        place_id = int(str(race_id)[4] + str(race_id)[5])
        race_num = int(str(race_id)[10] + str(race_id)[11])
        race_info = race_card.get_race_info(race_id)
        race_name = race_info[0]
        start_time = str(race_info[1]) + ":" + str(race_info[2])

        # 日付の出力
        f.write(str(race_day.year) + "/" + str(race_day.month) + "/" + str(race_day.day) + "\n")
        # 開催情報の出力
        f.write(name_header.NAME_LIST[place_id - 1] + str(race_num) + "R" + " " + race_name + " " + start_time + "\n\n")
        # 予想の出力
        for rank in range(5):
            if rank < len(pred_list):
                f.write(" " + name_header.SYMBOL_LIST[rank] + " " + str(pred_list[rank][0]) + " " + pred_list[rank][1] + "\n")
        f.write("\n\n")

        # タグの出力
        f.write("#MAR競馬予想\n")
        f.write("#競馬予想AI\n")
        f.write("#競馬 #競馬予想\n")
        f.write("#" + name_header.NAME_LIST[place_id - 1] + "競馬場\n")

        # # 9~12レースのみ名前を取得
        # if str(race_num) == "9" or str(race_num) == "10" or str(race_num) == "11" or str(race_num) == "12" :
        #     f.write("#" + race_name + "\n")
    
        f.close()
    except Exception as e:
        make_test_error(e)      

def make_return_text(place_id, race_day = date.today()):
    """当日の予想結果のテキスト作成
        Args:
            place_id(int) : place_id
            race_day(date) : レース開催日
    """
    try:
        # 今日のid_listを取得
        race_id_list = get_race_id.get_daily_id(place_id, race_day)
        
        # 的中率・回収率の計算
        if race_id_list :
            # テキストファイルの準備
            folder_path = name_header.TEXT_PATH + "race_returns/" + race_day.strftime("%Y%m%d")
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            text_data_path = folder_path + "//" + name_header.PLACE_LIST[place_id - 1] + "_pred_score.txt"
            f = open(text_data_path, "w", encoding = "UTF-8")

            # 日付の出力
            f.write(str(race_day.year) + "/" + str(race_day.month) + "/" + str(race_day.day) + " ")
            # 開催競馬場の出力
            f.write(name_header.NAME_LIST[place_id - 1] + "競馬場\n\n" + "MAR競馬予想 回収率\n")

            # 予想結果の出力
            write_win_hit_text(race_day, race_id_list, f)
            write_place_hit_text(race_day, race_id_list, f)
            write_trio_hit_text(race_day, race_id_list, f)     
            f.write("\n\n")

            # タグの出力
            f.write("#MAR競馬予想\n")
            f.write("#競馬予想AI\n")
            f.write("#競馬 #競馬予想\n")
            f.write("#" + name_header.NAME_LIST[place_id - 1] + "競馬場\n")
            f.close()
    except Exception as e:
        make_test_error(e)

if __name__ == "__main__":
    place_id = 5
    race_day = date.today() - timedelta(2)
    race_id_list = get_race_id.get_daily_id(place_id, race_day)
    for race_id in race_id_list:
        make_race_text(race_day, race_id)