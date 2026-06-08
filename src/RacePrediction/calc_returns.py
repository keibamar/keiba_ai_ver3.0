import math
import os
import re
import sys

from datetime import date, timedelta
import pandas as pd
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import get_race_id
import name_header
import post_text
import scraping

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
import race_card
import race_returns

import make_text

def calc_returns_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def save_day_race_return_csv(place_id, race_day, return_result_df):
    """一日の配当結果を保存
        Args:
            place_id(int) : place_id
            race_day(date) : レース開催日
            return_result_df(pd.DataFrame) : 配当結果
    """
    try:
        folder_path = name_header.TEXT_PATH + "race_returns/" + race_day.strftime("%Y%m%d")
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        text_data_path = folder_path + "//" +name_header.PLACE_LIST[place_id - 1] + "_pred_score.csv"
        return_result_df.to_csv(text_data_path)
    except Exception as e:
        calc_returns_error(e)

def save_each_race_return_csv(race_id, df_returns):
    """各レースの配当結果を保存
        Args:
            race_id(str) : race_id
            df_returns(pd.DataFrame) : 配当結果
    """
    try:
        place_id = int(race_id[4:6])
        year = int(race_id[0:4])
        folder_path = name_header.DATA_PATH + "RaceReturns/" +name_header.PLACE_LIST[place_id - 1] + "/" + str(year)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        csv_data_path = folder_path + "//" + str(race_id) + ".csv"
        df_returns.to_csv(csv_data_path)
    except Exception as e:
        calc_returns_error(e)

def get_each_race_retutn_csv(race_id):
    """配当結果csvの取得
        Args: 
            race_id(str) : race_id
        Returns:
            df_returns(pd.DataFrame) : 配当結果
    """
    try:
        place_id = int(race_id[4:6])
        year = int(race_id[0:4])
        folder_path = name_header.DATA_PATH + "RaceReturns/" +name_header.PLACE_LIST[place_id - 1] + "/" + str(year)
        csv_data_path = folder_path + "//" + str(race_id) + ".csv"
        df_returns = pd.read_csv(csv_data_path, index_col = 0, dtype = str)
        return df_returns
    except Exception as e:
        calc_returns_error(e)
        return pd.DataFrame() 

def add_symbol_in_selection(string, symbol, num):
    """ 式別の文字列に記号を付与する 
        Args:
            string(str) : 文字列
            symbol(str) : 記号
            num(int) : 記号を付与する空白の数
        Returns : 
            symbol_add_string(str) : 記号を付与した文字列 
    """
    space_indices = [i for i in range(len(string)) if string.startswith("br", i)]
    space_indices = [element for i, element in enumerate(space_indices) if (i + 1) % num != 0]
    symbol_add_string = ""
    for i in range(len(string)):
        symbol_add_string = symbol_add_string + string[i]
        if (i - 1) in space_indices:
            symbol_add_string = symbol_add_string + symbol + "br"
    # 不要な文字の消去
    symbol_add_string = symbol_add_string.replace("[", "").replace("]", "").replace("'", "")
    return symbol_add_string

def format_day_race_returns(df_return):
    """day_race_returnの整形
        Args:
            df_return(pd.DataFrame) : day_race_returnのデータフレーム
        Returns:
            format_df_return(pd.DataFrame) : 整形したデータフレーム

    """
    df_return = race_returns.format_race_return_dataframe(df_return)
    format_df_return = pd.DataFrame()
    format_df_return = pd.concat([format_df_return, df_return["単勝":"単勝"]])
    format_df_return = pd.concat([format_df_return, df_return["複勝":"複勝"]])

    if race_returns.is_bracket_quinella(df_return) :
        temp = df_return["枠連":"枠連"].reset_index().T
        temp.iloc[1,:] = add_symbol_in_selection(str(temp.iloc[1,:].values), "-", 2)
        format_df_return = pd.concat([format_df_return, temp.T.set_index(0)])
    
    temp = df_return["馬連":"馬連"].reset_index().T
    temp.iloc[1,:] = add_symbol_in_selection(str(temp.iloc[1,:].values), "-", 2)
    format_df_return = pd.concat([format_df_return, temp.T.set_index(0)])

    temp = df_return["ワイド":"ワイド"].reset_index().T
    temp.iloc[1,:] = add_symbol_in_selection(str(temp.iloc[1,:].values), "-", 2)
    format_df_return = pd.concat([format_df_return, temp.T.set_index(0)])

    temp = df_return["馬単":"馬単"].reset_index().T
    temp.iloc[1,:] = add_symbol_in_selection(str(temp.iloc[1,:].values), "→", 2)
    format_df_return = pd.concat([format_df_return, temp.T.set_index(0)])

    temp = df_return["3連複":"3連複"].reset_index().T
    temp.iloc[1,:] = add_symbol_in_selection(str(temp.iloc[1,:].values), "-", 3)
    format_df_return = pd.concat([format_df_return, temp.T.set_index(0)])

    temp = df_return["3連単":"3連単"].reset_index().T
    temp.iloc[1,:] = add_symbol_in_selection(str(temp.iloc[1,:].values), "→", 3)
    format_df_return = pd.concat([format_df_return, temp.T.set_index(0)])

    return format_df_return

def get_race_return(race_id):
    """配当結果の取得
        Args: 
            race_id(str) : race_id
        Returns:
            df_returns(pd.DataFrame) : 配当結果
    """
    # データフレームの整形
    df_return = scraping.scrape_day_race_returns(race_id)
    if df_return.empty:
        print(f"race_id '{race_id}' の配当結果が取得できませんでした")
        return pd.DataFrame()
    df_return = format_day_race_returns(df_return)
    # 配当結果の取得
    df_returns = pd.DataFrame()
    df_returns = pd.concat([df_returns, race_returns.format_type_returns_dataframe(df_return, "単勝")])
    df_returns = pd.concat([df_returns, race_returns.format_type_returns_dataframe(df_return, "複勝")])
    if race_returns.is_bracket_quinella(df_return) :
        df_returns = pd.concat([df_returns, race_returns.format_type_returns_dataframe(df_return, "枠連")])
    df_returns = pd.concat([df_returns, race_returns.format_type_returns_dataframe(df_return, "馬連")])
    df_returns = pd.concat([df_returns, race_returns.format_type_returns_dataframe(df_return, "ワイド")])
    df_returns = pd.concat([df_returns, race_returns.format_type_returns_dataframe(df_return, "馬単")])
    df_returns = pd.concat([df_returns, race_returns.format_type_returns_dataframe(df_return, "3連複")])
    df_returns = pd.concat([df_returns, race_returns.format_type_returns_dataframe(df_return, "3連単")])
    return df_returns.reset_index(drop = True)

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
        pred_df = race_card.get_race_cards(race_day, race_id)
        if not "rank" in pred_df.columns:
            print("not rank:" + str(race_id))
            race_num_diff += 1
            continue
        
        # 配当データの取得
        returns_df = get_each_race_retutn_csv(race_id)
        if returns_df.empty:
            race_num_diff += 1
            continue

        # 1着的中率・回収率
        win_df = returns_df[returns_df["式別"] == "単勝"].reset_index(drop = True)
        for i in range(len(win_df)):
            num = int(win_df.at[i,"馬番"])
            if pred_df.at[num - 1,"rank"] == 1:
                win_hit_rate += 1
                win_return_rate = win_return_rate + float(win_df.at[i,"配当"])
                win_hit_race += str(int(str(race_id)[10] + str(race_id)[11])) + " "
    
    if not len(race_id_list) == race_num_diff:
        win_hit_rate = (win_hit_rate / (len(race_id_list) - race_num_diff)) 
        win_return_rate = (win_return_rate / (len(race_id_list) - race_num_diff))
    return win_hit_rate, win_return_rate, win_hit_race

def get_place_result(race_day, race_id_list):
    """指数一位馬の複勝回収率・的中率・的中レースを出力
        Args:
           race_day(date) : レース開催日
           race_id_list(list) : 回収率を計算するrace_idのリスト
        Returns:
            win_hit_rate(int) : 的中率
            win_return_rate(int) : 回収率
            win_hit_race(str) : 的中レース
    """
    place_hit_rate = 0
    place_return_rate = 0
    place_hit_race = ""
    race_num_diff = 0
    for race_id in race_id_list:
        # 予想結果の取得
        pred_df = race_card.get_race_cards(race_day, race_id)
        if not "rank" in pred_df.columns:
            print("not rank:" + str(race_id))
            race_num_diff += 1
            continue
        
        # 配当データの取得
        returns_df = get_each_race_retutn_csv(race_id)
        if returns_df.empty:
            race_num_diff += 1
            continue
        
       # ３着内率・複勝回収率
        place_df = returns_df[returns_df["式別"] == "複勝"].reset_index(drop = True)
        for i in range(len(place_df)):
            num = int(place_df.at[i,"馬番"])
            if pred_df.at[num - 1, "rank"] == 1:
                place_hit_rate = place_hit_rate + 1
                if type(place_df.at[i,"配当"]) == str:
                    place_df.at[i,"配当"] = re.sub(r"\D","",place_df.at[i,"配当"])
                place_return_rate = place_return_rate + float(place_df.at[i,"配当"])
                place_hit_race += str(int(str(race_id)[10] + str(race_id)[11])) + " "
    
    if not len(race_id_list) == race_num_diff:
        place_hit_rate = (place_hit_rate / (len(race_id_list) - race_num_diff)) 
        place_return_rate = (place_return_rate / (len(race_id_list) - race_num_diff))
    return place_hit_rate, place_return_rate, place_hit_race

def get_quinella_box_result(race_day, race_id_list, box_num):
    """馬連BOXの回収率・的中率・的中レースを出力
        Args:
           race_day(date) : レース開催日
           race_id_list(list) : 回収率を計算するrace_idのリスト
           box_num(int) : ボックスの頭数
        Returns:
            quinella_box_hit_rate(int) : 的中率
            quinella_box_return_rate(int) : 回収率
            quinella_box_hit_race(str) : 的中レース
    """
    quinella_box_hit_rate = 0
    quinella_box_return_rate = 0
    quinella_box_hit_race = ""
    race_num_diff = 0

    for race_id in race_id_list:
        # 予想結果の取得
        pred_df = race_card.get_race_cards(race_day, race_id)
        if not "rank" in pred_df.columns:
            print("not rank:" + str(race_id))
            race_num_diff += 1
            continue
        
        # 配当データの取得
        returns_df = get_each_race_retutn_csv(race_id)
        if returns_df.empty:
            race_num_diff += 1
            continue
        
       # 馬連BOX的中率・回収率
        quinella_df = returns_df[returns_df["式別"] == "馬連"].reset_index(drop = True)
        for i in range(len(quinella_df)):
             # 馬連の馬番の取得
            num_list = re.findall(r"\d+", quinella_df.at[i,"馬番"])
            # １着馬・２着馬の予想順位を取得
            rank_1 = pred_df[pred_df["馬番"] == int(num_list[0])].reset_index(drop = True).at[0 ,"rank"]
            rank_2 = pred_df[pred_df["馬番"] == int(num_list[1])].reset_index(drop = True).at[0 ,"rank"]
            
            # ランキング３位以内の２頭の場合
            eval_1 = rank_1  <= box_num
            eval_2 = rank_2  <= box_num
            
            if eval_1 and eval_2:
                quinella_box_hit_rate +=  1
                if type(quinella_df.at[i, "配当"]) == str:
                    quinella_df.at[i, "配当"] = re.sub(r"\D","",quinella_df.at[i, "配当"])
                quinella_box_return_rate += float(quinella_df.at[i, "配当"])
                quinella_box_hit_race += str(int(str(race_id)[10] + str(race_id)[11])) + " "

    if not len(race_id_list) == race_num_diff:
        quinella_box_hit_rate = (quinella_box_hit_rate / (len(race_id_list) - race_num_diff)) 
        quinella_box_return_rate = (quinella_box_return_rate /(len(race_id_list) - race_num_diff)) / (math.comb(box_num,2))
    return quinella_box_hit_rate, quinella_box_return_rate, quinella_box_hit_race

def get_quinella_wheel_result(race_day, race_id_list, wheel_num):
    """馬連流しの回収率・的中率・的中レースを出力
        Args:
           race_day(date) : レース開催日
           race_id_list(list) : 回収率を計算するrace_idのリスト
           wheel_num(int) : 流し数
        Returns:
            quinella_wheel_hit_rate(int) : 的中率
            quinella_wheel_return_rate(int) : 回収率
            quinella_wheel_hit_race(str) : 的中レース
    """
    quinella_wheel_hit_rate = 0
    quinella_wheel_return_rate = 0
    quinella_wheel_hit_race = ""
    race_num_diff = 0

    for race_id in race_id_list:
        # 予想結果の取得
        pred_df = race_card.get_race_cards(race_day, race_id)
        if not "rank" in pred_df.columns:
            print("not rank:" + str(race_id))
            race_num_diff += 1
            continue
        
        # 配当データの取得
        returns_df = get_each_race_retutn_csv(race_id)
        if returns_df.empty:
            race_num_diff += 1
            continue
        
       # 馬連流し的中率・回収率
        quinella_df = returns_df[returns_df["式別"] == "馬連"].reset_index(drop = True)
        for i in range(len(quinella_df)):
             # 馬連の馬番の取得
            num_list = re.findall(r"\d+", quinella_df.at[i,"馬番"])
            # １着馬・２着馬の予想順位を取得
            rank_1 = pred_df[pred_df["馬番"] == int(num_list[0])].reset_index(drop = True).at[0 ,"rank"]
            rank_2 = pred_df[pred_df["馬番"] == int(num_list[1])].reset_index(drop = True).at[0 ,"rank"]
            
            # 軸流し
            if (rank_1 == 1 and rank_2 <= (wheel_num + 1)) or (rank_1 <= (wheel_num + 1) and rank_2 == 1):
                quinella_wheel_hit_rate +=  1
                if type(quinella_df.at[i, "配当"]) == str:
                    quinella_df.at[i, "配当"] = re.sub(r"\D","",quinella_df.at[i, "配当"])
                quinella_wheel_return_rate += float(quinella_df.at[i, "配当"])
                quinella_wheel_hit_race += str(int(str(race_id)[10] + str(race_id)[11])) + " "

    if not len(race_id_list) == race_num_diff:
        quinella_wheel_hit_rate = (quinella_wheel_hit_rate / (len(race_id_list) - race_num_diff)) 
        quinella_wheel_return_rate = (quinella_wheel_return_rate /(len(race_id_list) - race_num_diff)) / (wheel_num)
    return quinella_wheel_hit_rate, quinella_wheel_return_rate, quinella_wheel_hit_race

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
        pred_df = race_card.get_race_cards(race_day, race_id)
        if not "rank" in pred_df.columns:
            print("not rank:" + str(race_id))
            race_num_diff += 1
            continue
        
        # 配当データの取得
        returns_df = get_each_race_retutn_csv(race_id)
        if returns_df.empty:
            race_num_diff += 1
            continue
        
       # 三連複BOX的中率・回収率
        trio_df = returns_df[returns_df["式別"] == "3連複"].reset_index(drop = True)
        for i in range(len(trio_df)):
            # 三連複の馬番の取得
            num_list = re.findall(r"\d+", trio_df.at[i,"馬番"])
            # １着馬・２着馬・３着馬の予想順位を取得
            rank_1 = pred_df[pred_df["馬番"] == int(num_list[0])].reset_index(drop = True).at[0 ,"rank"]
            rank_2 = pred_df[pred_df["馬番"] == int(num_list[1])].reset_index(drop = True).at[0 ,"rank"]
            rank_3 = pred_df[pred_df["馬番"] == int(num_list[2])].reset_index(drop = True).at[0 ,"rank"]

            # ランキング５位以内で３頭の場合
            eval_1 = rank_1  <= box_num
            eval_2 = rank_2  <= box_num
            eval_3 = rank_3  <= box_num
            if eval_1 and eval_2 and eval_3:
                trio_box_hit_rate +=  1
                if type(trio_df.at[i, "配当"]) == str:
                    trio_df.at[i, "配当"] = re.sub(r"\D","",trio_df.at[i, "配当"])
                trio_box_return_rate += float(trio_df.at[i, "配当"])
                trio_box_hit_race += str(int(str(race_id)[10] + str(race_id)[11])) + " "
    if not len(race_id_list) == race_num_diff:
        trio_box_hit_rate = (trio_box_hit_rate / (len(race_id_list) - race_num_diff)) 
        trio_box_return_rate = (trio_box_return_rate /(len(race_id_list) - race_num_diff)) / (math.comb(box_num,3))
    return trio_box_hit_rate, trio_box_return_rate, trio_box_hit_race

def get_trio_wheel_result(race_day, race_id_list, wheel_num):
    """三連複流しの回収率・的中率・的中レースを出力
        Args:
            race_day(date) : レース開催日
            race_id_list(list) : 回収率を計算するrace_idのリスト
            wheel_num(int) : 流し数
        Returns:
            trio_wheel_hit_rate(int) : 的中率
            trio_wheel_return_rate(int) : 回収率
            trio_wheel_hit_race(str) : 的中レース
    """
    trio_wheel_hit_rate = 0
    trio_wheel_return_rate = 0
    trio_wheel_hit_race = ""
    race_num_diff = 0
    for race_id in race_id_list:
        # 予想結果の取得
        pred_df = race_card.get_race_cards(race_day, race_id)
        if not "rank" in pred_df.columns:
            print("not rank:" + str(race_id))
            race_num_diff += 1
            continue
        
        # 配当データの取得
        returns_df = get_each_race_retutn_csv(race_id)
        if returns_df.empty:
            race_num_diff += 1
            continue
        
       # 三連複的中率・回収率(上位5頭BOX)
        trio_df = returns_df[returns_df["式別"] == "3連複"].reset_index(drop = True)
        for i in range(len(trio_df)):
            # 三連複の馬番の取得
            num_list = re.findall(r"\d+", trio_df.at[i,"馬番"])
            # １着馬・２着馬・３着馬の予想順位を取得
            rank_1 = pred_df[pred_df["馬番"] == int(num_list[0])].reset_index(drop = True).at[0 ,"rank"]
            rank_2 = pred_df[pred_df["馬番"] == int(num_list[1])].reset_index(drop = True).at[0 ,"rank"]
            rank_3 = pred_df[pred_df["馬番"] == int(num_list[2])].reset_index(drop = True).at[0 ,"rank"]

            if (rank_1 == 1 and rank_2 <= (wheel_num + 1) and rank_3 <= (wheel_num + 1)) \
            or (rank_1 <= (wheel_num + 1) and rank_2 == 1 and rank_3 <= (wheel_num + 1)) \
            or (rank_1 <= (wheel_num + 1) and rank_2 <= (wheel_num + 1) and rank_3 == 1):
                trio_wheel_hit_rate +=  1
                if type(trio_df.at[i, "配当"]) == str:
                    trio_df.at[i, "配当"] = re.sub(r"\D","",trio_df.at[i, "配当"])
                trio_wheel_return_rate += float(trio_df.at[i, "配当"])
                trio_wheel_hit_race += str(int(str(race_id)[10] + str(race_id)[11])) + " "
    if not len(race_id_list) == race_num_diff:
        trio_wheel_hit_rate = (trio_wheel_hit_rate / (len(race_id_list) - race_num_diff)) 
        trio_wheel_return_rate = (trio_wheel_return_rate /(len(race_id_list) - race_num_diff)) / (math.comb(wheel_num, 2))
    return trio_wheel_hit_rate, trio_wheel_return_rate, trio_wheel_hit_race

def calc_day_race_return(place_id, race_day, race_id_list):
    """1日の回収率・的中率を計算
        Args:
            place_id(int) : place_id
            race_day(date) : レース開催日
            race_id_list(list) : 回収率を計算するrace_idのリスト
    """
    column = ["的中率", "回収率", "的中レース"]
    # 単勝回収率・的中率
    win_hit_rate, win_return_rate, win_hit_race = get_win_result(race_day, race_id_list)
    win_return_result = pd.DataFrame([[win_hit_rate, win_return_rate, win_hit_race]], index = ["単勝"], columns = column)
    
    # 複勝回収率・的中率
    place_hit_rate, place_return_rate, place_hit_race = get_place_result(race_day, race_id_list)
    place_return_result = pd.DataFrame([[place_hit_rate, place_return_rate, place_hit_race]], index = ["複勝"], columns = column)
    
    # 馬連回収率・的中率
    quinella_3box_hit_rate, quinella_3box_return_rate, quinella_3box_hit_race = get_quinella_box_result(race_day, race_id_list, 3)
    quinella_return_result = pd.DataFrame([[quinella_3box_hit_rate, quinella_3box_return_rate, quinella_3box_hit_race]], index = ["馬連3頭BOX"], columns = column)
    quinella_5box_hit_rate, quinella_5box_return_rate, quinella_5box_hit_race = get_quinella_box_result(race_day, race_id_list, 5)
    quinella_return_result = pd.concat([quinella_return_result, pd.DataFrame([[quinella_5box_hit_rate, quinella_5box_return_rate, quinella_5box_hit_race]], index = ["馬連5頭BOX"], columns = column)])
    quinella_3wheel_hit_rate, quinella_3wheel_return_rate, quinella_3wheel_hit_race = get_quinella_wheel_result(race_day, race_id_list, 3)
    quinella_return_result = pd.concat([quinella_return_result, pd.DataFrame([[quinella_3wheel_hit_rate, quinella_3wheel_return_rate, quinella_3wheel_hit_race]], index = ["馬連3頭流し"], columns = column)])
    quinella_5wheel_hit_rate, quinella_5wheel_return_rate, quinella_5wheel_hit_race = get_quinella_wheel_result(race_day, race_id_list, 5)
    quinella_return_result = pd.concat([quinella_return_result, pd.DataFrame([[quinella_5wheel_hit_rate, quinella_5wheel_return_rate, quinella_5wheel_hit_race]], index = ["馬連5頭流し"], columns = column)])
    
    # 3連複回収率・的中率
    trio_3box_hit_rate, trio_3box_return_rate, trio_3box_hit_race = get_trio_box_result(race_day, race_id_list, 3)
    trio_return_result = pd.DataFrame([[trio_3box_hit_rate, trio_3box_return_rate, trio_3box_hit_race]], index = ["3連複3頭BOX"], columns = column)
    trio_5box_hit_rate, trio_5box_return_rate, trio_5box_hit_race = get_trio_box_result(race_day, race_id_list, 5)
    trio_return_result = pd.concat([trio_return_result, pd.DataFrame([[trio_5box_hit_rate, trio_5box_return_rate, trio_5box_hit_race]], index = ["3連複5頭BOX"], columns = column)])
    trio_3wheel_hit_rate, trio_3wheel_return_rate, trio_3wheel_hit_race = get_trio_wheel_result(race_day, race_id_list, 3)
    trio_return_result = pd.concat([trio_return_result, pd.DataFrame([[trio_3wheel_hit_rate, trio_3wheel_return_rate, trio_3wheel_hit_race]], index = ["3連複1軸3頭流し"], columns = column)])
    trio_5wheel_hit_rate, trio_5wheel_return_rate, trio_5wheel_hit_race = get_trio_wheel_result(race_day, race_id_list, 5)
    trio_return_result = pd.concat([trio_return_result, pd.DataFrame([[trio_5wheel_hit_rate, trio_5wheel_return_rate, trio_5wheel_hit_race]], index = ["3連複1軸5頭流し"], columns = column)])
    
    # データフレームに格納
    return_result_df = pd.DataFrame()
    return_result_df = pd.concat([return_result_df, win_return_result])
    return_result_df = pd.concat([return_result_df, place_return_result])
    return_result_df = pd.concat([return_result_df, quinella_return_result])
    return_result_df = pd.concat([return_result_df, trio_return_result])

    # テキストファイルの準備
    save_day_race_return_csv(place_id, race_day, return_result_df)    

def calc_day_race_return_all(race_day = date.today()):
    """指定日の全開催場回収率・的中率を計算
        Args:
            race_day(Date) : 日（初期値：今日）
    """
    # 今日のid_listを取得
    race_id_list = get_race_id.get_daily_id(race_day = race_day)

    # 各競馬場ごとにrace_idを分割
    course_list = []
    for i in range(0, len(race_id_list), 12):
        course_list.append(race_id_list[i: i + 12])

    # 各競馬場毎の的中率・回収率の計算
    for race_id_list in course_list:
        print(race_id_list)
        place_id = int(str(race_id_list[0])[4] + str(race_id_list[0])[5])
        calc_day_race_return(place_id, race_day, race_id_list)


def post_race_rerurns(place_id, race_day):
    """レースの配当結果をポスト
        Args:
            place_id(int) : place_id
            race_day(date) : レース開催日
    """
    # テキストファイルの読み込み
    text_data_path =  name_header.TEXT_PATH + "race_returns/" + race_day.strftime("%Y%m%d") + "//" + name_header.PLACE_LIST[place_id - 1] + "_pred_score.txt"
    post_text.post_text_data(text_data_path)

def save_calc_day_return(race_day = date.today()):
    """指定日の全開催場回収率・的中率を計算して保存
    """ 
    # 今日のid_listを取得
    race_id_list = get_race_id.get_daily_id(race_day = race_day)
    if not race_id_list:
        return
    # 各競馬場ごとにrace_idを分割
    course_list = []
    for i in range(0, len(race_id_list), 12):
        course_list.append(race_id_list[i: i + 12])     
    # 各競馬場毎の的中率・回収率の計算
    for race_id_list in course_list:
        print(race_id_list)
        for race_id in race_id_list:
            # データフレームの整形
            df_return = get_race_return(race_id)
            if df_return.empty:
                continue
            # 各レースの配当結果を保存
            print(df_return)
            save_each_race_return_csv(race_id, df_return)

def run_all_year(year):
    """指定した年の全日付で save_calc_day_return() を実行（今日より後は実行しない）"""
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    today = date.today()

    current = start
    while current <= end and current <= today:
        try:
            print(f"{current}")
            save_calc_day_return(current)
        except Exception as e:
            print(f"{current} の処理中にエラー: {e}")
        current += timedelta(days=1)

def weekly_update_race_returns(race_day = date.today()):
    race_id_list = get_race_id.get_past_weekly_id(base_day = race_day)
    for race_id in race_id_list:
        print(race_id)

        # データフレームの整形
        df_return = get_race_return(race_id)
        if df_return.empty:
            print("Empty Race Return : ", race_id)
            continue

        # 各レースの配当結果を保存
        save_each_race_return_csv(race_id, df_return)
        

if __name__ =="__main__":
    race_day = date.today()
    calc_day_race_return_all(race_day)
   
    for place_id in get_race_id.get_daily_place_id(race_day):
        make_text.make_return_text(place_id, race_day)
        post_race_rerurns(place_id, race_day)
