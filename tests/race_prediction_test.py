import re
import sys

import datetime
from datetime import date, timedelta, datetime
from time import sleep
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import get_race_id
import name_header
import post_text
import string_format
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\RacePrediction")
import make_text
import race_card
import day_race_prediction
import calc_returns

def make_race_card_test():
    return

def calc_race_return_test():
    return

def format_datetime(date_string):
    """レース開催日かチェックする
        Args:
            date_string(str) : 日付
        Returns:
            True/False
    """
    # 年、月、日を抽出
    year = int(date_string[:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])
    
    # datetimeオブジェクトに変換
    race_day = datetime(year, month, day)
    return race_day

if __name__ == "__main__":

    while(1):
        # print("[0]予想のみ/[1]予想+配当結果")
        # test_type = input(">")
        # if (not test_type == "0") and (not test_type == "1") :
        #     string_format.format_error(test_type)
        #     continue

        # 日付の取得
        reset_day_flag = 0
        race_id_lists = []
        while(1):
            print("日付を指定してください。[yyyymmdd]")
            print("複数の日付を指定する場合は「,」で区切って入力してください。")
            str_race_day_input = input(">")
            str_race_day_input = str_race_day_input.replace(" ", "")
            str_race_day_list = str_race_day_input.split(",")

            for str_race_day in str_race_day_list :
                if not string_format.is_valid_date_format(str_race_day):
                    reset_day_flag = 1
                    break
                race_id_list = get_race_id.get_daily_id(race_day = format_datetime(str_race_day))
                if not any(race_id_list):
                    print(str_race_day, "に開催されたレースはありません。")
                    str_race_day_list.remove(str_race_day)
                else:
                    race_id_lists.append(race_id_list)
            
            if reset_day_flag == 1 :
                print("再度日付を指定しますか。[y/n]")
                reset_day= input(">")
                if reset_day == "y" or reset_day == "Y":
                    race_id_lists = []
                    reset_day_flag = 0
                    continue
                else:
                    break
            else:
                break
        # 再度日付を指定しない場合、処理を終了する
        if reset_day_flag == 1 :
            break
        
        for id, str_race_day in enumerate(str_race_day_list):
            print(str_race_day, "の開催場は以下になります。どの開催場の予想を出力しますか。")
            place_id_list = get_race_id.get_daily_place_id(race_day = format_datetime(str_race_day))
            while(1):
                print("[0] 全ての競馬場")
                for i, place_id, in enumerate(place_id_list):
                    print(f"[{i + 1}] {name_header.NAME_LIST[place_id - 1]}競馬場")
                
                pred_course = input(">")
                if not( "0" <= pred_course <=f"{len(place_id_list)}") :
                    string_format.format_error(pred_course)
                    continue
                break
            
            # すべての競馬場指定時は全レース予想
            if pred_course == "0":
                race_card.daily_race_card(place_id = 0, race_day = format_datetime(str_race_day))
            else :
                place_id = place_id_list[int(pred_course) - 1]
                print("予想するレースを指定してください。")            
                while(1):
                    print("[0] 全レース予想")
                    print("[1] レースを指定する")
                    race_type = input(">")
                    if not( "0" <= race_type <= "1") :
                        string_format.format_error(pred_course)
                        continue
                    break

                if race_type == "0":
                    race_card.daily_race_card(place_id = place_id, race_day = format_datetime(str_race_day))
                elif race_type == "1":
                    print("レースを指定してください。[1~12]")
                    print("複数のレースを指定する場合は「,」で区切って入力してください。")
                    race_num_inupt = input(">")
                    race_num_inupt = race_num_inupt.replace(" ", "")
                    race_num_list = race_num_inupt.split(",")
                    
                    # レースidの取得
                    race_id_list = race_id_lists[id]
                    for race_num in race_num_list:
                        if not( 1 <= int(race_num) <= 12) :
                            string_format.format_error(race_num)
                            continue
                        race_id = race_id_list[12 * ( int(pred_course) - 1 ) + int(race_num) - 1]
                        print("Make Race Card", race_id)
                        race_card_df = race_card.make_race_card(race_id = race_id)
                        # csvファイルで出力
                        race_card.save_race_cards(race_card_df, race_day = format_datetime(str_race_day), race_id = race_id)




        break

    