
import sys

import datetime
from datetime import date, timedelta
from time import sleep
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\web\src")
import name_header
import post_text
import mail_api

import analysis_race_info
import make_time_id_list
import make_text
import race_card
import calc_returns

import daily_race_results

from generators.date_index import add_race_day
from generators.daily_index import make_daily_index_page
from generators.race_pages import make_race_card_html  

def post_race_pred(race_id, race_day):
    """レース予想のポスト
        Args:
            race_id(int) : race_id
            race_day(date) : レース開催日
    """
    # テキストファイルの読み込み
    text_path = name_header.TEXT_PATH + "race_prediction/" + race_day.strftime("%Y%m%d") + "/" + str(race_id) + ".txt"
    print(text_path)
    post_text.post_text_data(text_path)

def post_pred_return(place_id, race_day):
    """予想結果の配当のポスト
        Args:
            place_id(int) : place_id
            race_day(date) : レース開催日
    """
    # テキストファイルの読み込み
    text_path = name_header.TEXT_PATH + "race_returns/" + race_day.strftime("%Y%m%d") + "/" + name_header.PLACE_LIST[place_id - 1] + "_pred_score.txt"
    post_text.post_text_data(text_path)

def is_post_race(race_id):
    """ポストするレースか判定する
        Args:
            race_id: race_id
        Returns:
            bool : ポストするか否か
    """
    place_id = int(str(race_id)[4] + str(race_id)[5])
    race_num = int(str(race_id)[10] + str(race_id)[11])
    for post_race_num in name_header.POST_RACE_LIST[place_id - 1]:
        if race_num == post_race_num:
            return True
    return False


def post_daily_race_pred(race_day = date.today()):
    """一日のレースの予想をポスト
        Args:
            race_day(date) : レース開催日(初期値:今日)
    """
    date_str = race_day.strftime("%Y%m%d")
    time_id_list = make_time_id_list.get_time_id_list(race_day)
    add_race_day(race_day)
    make_daily_index_page(race_day)
    # 過去一週間のindexを再作成（リンクの生成）
    for delta_day in range(1, 8):
        past_day = race_day - timedelta(days=delta_day)
        make_daily_index_page(past_day)

    # place_id 毎に直前に処理した race_id を保持する
    last_race_by_place = {}

    while(any(time_id_list)):
        # レース20分前に投稿
        comp_time = datetime.datetime.now() + timedelta(minutes=20)
        str_comp_time = str(comp_time.hour).zfill(2) + str(comp_time.minute).zfill(2)
        race_time = time_id_list[0][0]
        # 実行時間を過ぎていたら投稿を実行
        if(int(race_time) <= (int(str_comp_time))):
           race_id = time_id_list[0][1]
           try:
                # 予想の更新
                race_card_df, race_info_df = race_card.make_race_card(race_id)
                # csvファイルで出力
                race_card.save_race_cards(race_card_df, race_day, race_id)
                race_card.save_race_info_df(race_info_df, race_day, race_id)
                analysis_race_info.update_horse_name_id_map(race_card_df)
                # textの作成
                make_text.make_race_text(race_day, race_id)
                # API対策で計12レースのみ投稿
                if len(time_id_list) <= 12:
                # if is_post_race(race_id):
                    post_race_pred(race_id, race_day)
                    print("post:" + str(race_time + ":" + str(race_id)))
                else :
                    print("no post for API restricctinos")
                # mail送信
                mail_api.send_race_pred(race_day, race_id)
           except :
               print("post_error:" + str(race_time + ":" + str(race_id)))
               print(sys.exc_info())

           #htmlの作成
           place_id = int(str(race_id)[4] + str(race_id)[5])
           print("make html:" + str(race_id))
           make_race_card_html(date_str, place_id, race_id)
           make_daily_index_page(race_day)

           # 直前の race_id を取得（存在すれば previous）
           previous_race_id = last_race_by_place.get(place_id)
           # 直前レースがあれば、結果の取得とhtmlを再生成(リンク更新のため)
           if previous_race_id:
               # レース結果の取得
                try :
                    results_df = daily_race_results.get_each_race_results(previous_race_id)
                    if not results_df.empty:
                        daily_race_results.save_each_race_result_csv(previous_race_id, results_df)
                except :
                    print("Miss Make Results : ", previous_race_id)
               # 配当結果の取得
                try :
                    df_return = calc_returns.get_race_return(previous_race_id)
                    if not df_return.empty:
                        calc_returns.save_each_race_return_csv(previous_race_id, df_return)
                except :
                    print("Miss Make Returns : ", previous_race_id)
                print("previous race html make:" + str(previous_race_id))    
                make_race_card_html(date_str, place_id, previous_race_id)
           # 今回処理した race_id を last_race_by_place に記録
           last_race_by_place[place_id] = race_id
           time_id_list.pop(0)
        
        # 1分ごとに実行
        sleep(60)
    
    # 最後のレースから30分待つ
    sleep(1800)
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        last_race_id = last_race_by_place.get(place_id)
        if last_race_id is not None:
            # レース結果の取得
            results_df = daily_race_results.get_each_race_results(last_race_id)
            if not results_df.empty:
                daily_race_results.save_each_race_result_csv(last_race_id, results_df)
            # 配当結果の取得
            df_return = calc_returns.get_race_return(last_race_id)
            if not df_return.empty:
                calc_returns.save_each_race_return_csv(last_race_id, df_return)
            
            print("previous race html make:" + str(last_race_id))    
            make_race_card_html(date_str, place_id, last_race_id)

if __name__ == '__main__':
    post_daily_race_pred()