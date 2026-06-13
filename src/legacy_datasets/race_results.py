import os
import re
import sys

from datetime import date, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm

# pycache を生成しない
sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import get_race_id
import name_header
import scraping

def race_results_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

# --- 重複列の自動処理 ---
def clean_columns(df):
    # 重複列がある場合にカウント付きでリネームする
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index
        for i, idx in enumerate(dup_idx):
            cols[idx] = f"{dup}_{i+1}"
    df.columns = cols
    return df
    
def scrape_race_results_dataframe(race_id_list):
    """ race_id_listのrace_resultsのDataFrameを作成 
        Args:
            race_id_list list(str) : race_idのリスト
        Returns:
            DataFrame : race_id_listのrace_results    
    """

    # スクレイピング
    race_results_df = pd.DataFrame()
    for race_id in tqdm(race_id_list):
        new_df = scraping.scrape_race_results(race_id)
        new_df = clean_columns(new_df)
        base_columns = list(dict.fromkeys(list(race_results_df.columns) + list(new_df.columns)))
        # 列の順番をそろえる
        new_df = new_df.reindex(columns=base_columns)
        # 結合（インデックスも無視して安全に）
        race_results_df = pd.concat([race_results_df, new_df], axis=0)
    return race_results_df

def format_race_results_dataframe(race_results_df):
    """ race_resultsのフォーマットを整える 
        Args:
            race_results_df（pd.DataFrame） : race_resultデータセット
        Return:
            race_results_df（pd.DataFrame） : フォーマットされたrace_resultデータセット
    """
    # 欠損値をnanで埋める
    race_results_df.fillna(np.nan)   
    # "斤量","人気"をfloatに固定
    # race_results_df['人気'] = race_results_df['人気'].astype(float)
    # race_results_df['斤量'] = race_results_df['斤量'].astype(float)
    # 文字列に変換
    race_results_df = race_results_df.astype(str)
    # 重複している内容を消去
    race_results_df = race_results_df.drop_duplicates(keep = 'first')

    return race_results_df

def save_race_results_dataset(place_id, year, race_results_df):
    """ race_resultsのDataFrameを保存 
        Args:
            place_id (int) : 開催コースid
            year(int) : 開催年
            race_results_df（pd.DataFrame） : race_resultデータセット
    """
    try:
        if any(race_results_df):
            # フォーマットを整える
            race_results_df = format_race_results_dataframe(race_results_df)
            # csv/pickleに保存
            race_results_df.to_csv(name_header.DATA_PATH + "RaceResults\\" + name_header.PLACE_LIST[place_id - 1] + '//' + str(year) + '_race_results.csv')
            race_results_df.to_pickle(name_header.DATA_PATH + "RaceResults\\" + name_header.PLACE_LIST[place_id - 1] + '//' + str(year) + '_race_results.pickle')
    except Exception as e:
            race_results_error(e)

def get_race_results_csv(place_id, year):
    """ race_resultsのcsvを取得する 
        Args:
            place_id (int) : 開催コースid
            year(int) : 開催年
    """
    # csvを読み込む 
    path = name_header.DATA_PATH + "RaceResults\\" + name_header.PLACE_LIST[place_id - 1] +  '//' + str(year) + '_race_results.csv'
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col = 0, dtype = str)
        df.fillna(np.nan)  
    else :
        df = pd.DataFrame()

    return df

def get_race_id_result(race_id):
    """race_idのレース結果の取得
        Args:
            race_id(str) : race_id
        Returns:
            race_results_df(pd.DataFrame) : レース結果のDataFrame
    """
    # race_idからyear, place_idの抽出
    year = int(str(race_id)[0] + str(race_id)[1] + str(race_id)[2] + str(race_id)[3])
    place_id = int(str(race_id)[4] + str(race_id)[5])

    # race_resultsの読出し
    df_results = get_race_results_csv(place_id, year)
    df_results.index = df_results.index.astype(str)
    # race_idのデータのみ抽出
    df_result = df_results[race_id:race_id]
    df_result = df_result.reset_index(drop = True)

    return df_result

def get_course_info(result):
    """レース情報を取得
        Args:
            result(pd.DataFrame) : race_resultのデータセット
        Returns:
            course_info : place_id, race_type, course_len, ground_state, race_class
    """
    # 開催競馬場を取得 (JRA以外は-1)
    place_id = -1
    course = result["開催"]
    for i in range(len(name_header.NAME_LIST)):
        if name_header.NAME_LIST[i] in course:
            place_id = i + 1
    if place_id < 0:
        return[place_id, " "," "," "," "]
    # レース情報を取得
    race_info = result["course_len"]
    if "芝" in race_info:
        race_type = "芝"
    elif "ダ" in race_info:
        race_type = "ダート"
    else:
        race_type = "障害"
    course_len = re.sub(r"\D","",race_info)

    # 馬場状態を取得
    ground_state = result["ground_state"]
    if ground_state in "稍":
        ground_state = "稍重"
    if ground_state in "不":
        ground_state = "不良"

    # クラスを取得
    race_class = result["class"]
    # race_name = result["レース名"]
    # if "新馬" in race_name:
    #     race_class = "新馬"
    # elif "未勝利" in race_name:
    #     race_class = "未勝利"
    # elif "1勝クラス" in race_name or "１勝クラス" in race_name:
    #     race_class = "1勝クラス"
    # elif "2勝クラス" in race_name or "２勝クラス" in race_name:
    #     race_class = "2勝クラス"
    # elif "3勝クラス" in race_name or "３勝クラス" in race_name:
    #     race_class = "3勝クラス"
    # else :
    #     race_class = "オープン"

    return [place_id, race_type, course_len, ground_state, race_class]

def export_race_info_per_race(place_id, year):
    """
    各レースIDごとに基本情報(race_type, course_len, weather, ground_state, class)を抽出して
    {year}/{race_id}.csv に保存する。

    Parameters
    ----------
    input_csv : str
        元のレース結果CSVファイルのパス
    output_base_path : str
        出力先のベースディレクトリ（例: "./race_info"）
    """
    input_csv = name_header.DATA_PATH + "RaceResults//" + name_header.PLACE_LIST[place_id - 1] + f"//{year}_race_results.csv"
    
    # CSV読み込み
    if not os.path.exists(input_csv):
        print("not file exists:", input_csv)
        return
    df = pd.read_csv(input_csv, dtype=str)
    if df.empty:
        print("⚠️ 入力CSVが空です。処理を終了します。")
        return

    out_dir = name_header.DATA_PATH + "RaceInfo//" + name_header.PLACE_LIST[place_id - 1] + "//" + str(year)
    os.makedirs(out_dir, exist_ok=True)

    # --- race_id 列の特定 ---
    if "race_id" in df.columns:
        pass  # そのまま使用
    elif "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "race_id"})
    else:
        # indexから復元（ファイルによってはインデックスがrace_idの場合もある）
        df = df.reset_index().rename(columns={"index": "race_id"})
    # race_id 列を文字列として確保
    df["race_id"] = df.index if "race_id" not in df.columns else df["race_id"]

    # 重複レースIDを除去（1レースにつき1行だけ）
    unique_race_ids = df["race_id"].unique()

    # 各レースごとにファイル作成
    for race_id in unique_race_ids:
        sub = df[df["race_id"] == race_id].iloc[0]  # 最初の1行だけ取得

        # 出力ファイルパス
        out_path = os.path.join(out_dir, f"{race_id}.csv")

        # 必要列を抽出
        record = {
            "race_type": sub.get("race_type", ""),
            "course_len": sub.get("course_len", ""),
            "weather": sub.get("weather", ""),
            "ground_state": sub.get("ground_state", ""),
            "class": sub.get("class", "")
        }

        # DataFrameとして保存
        out_df = pd.DataFrame([record])
        try :
            out_df["course_len"] = int(out_df["course_len"].astype(float))
        except Exception as e:
            out_df["course_len"] = out_df["course_len"]
        
        out_df.to_csv(out_path, index_label="", encoding="utf-8-sig")

        # print(f"✅ {out_path} を作成しました。")

    print(name_header.PLACE_LIST[place_id - 1], str(year), "レース情報出力完了")

def _to_int(val):
            if pd.isna(val) or val == "":
                return val
            s = str(val)
            # 数字を抽出して整数化（例: '1400m' -> 1400, '1400.0' -> 1400）
            m = re.search(r"(\d+)", s)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
            try:
                return int(float(s))
            except Exception:
                return val
            
def convert_course_len_csv(file_path: str) -> bool:
    """指定したCSVファイルに `course_len` 列があれば、すべて整数型に変換して上書き保存する。

    Args:
        file_path (str): 対象のCSVファイルパス

    Returns:
        bool: 変換と保存が成功したら True、ファイルが存在しない・列がない等の場合は False
    """
    try:
        if not os.path.isfile(file_path):
            print("convert_course_len_csv: file not exists:", file_path)
            return False

        df = pd.read_csv(file_path, dtype=str)
        if "course_len" not in df.columns:
            return False

        df["course_len"] = df["course_len"].apply(_to_int)

        # 上書き保存（UTF-8 BOM付きでエクセルでも開きやすく）
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        return True
    except Exception as e:
        race_results_error(e)
        return False

def update_race_results_dataset(place_id, day = date.today()):
    """ 開催コースと日にちを指定して、過去1週間分のrace_resultsデータセットを更新する 
        Args:
            place_id (int) : 開催コースid
            day(int) : 日（初期値：今日）
    """
    # race_id_listの取得
    race_id_list = get_race_id.get_past_weekly_id(place_id, day)

    # 過去のデータセットを取得
    old_race_results_df = get_race_results_csv(place_id, day.year)

    # レース結果の取得
    new_race_results_df = scrape_race_results_dataframe(race_id_list)

    # 更新するデータセットがあれば更新
    if any(new_race_results_df):
        try:
            if not old_race_results_df.empty :
                base_columns = old_race_results_df.columns
               # 列の順番をそろえる
                new_race_results_df = new_race_results_df.reindex(columns=base_columns)
            # 空文字や空白を NaN に変換
            new_race_results_df = new_race_results_df.replace(r'^\s*$', np.nan, regex=True)
            new_race_results_df.fillna(np.nan) 
            new_race_results_df = pd.concat([old_race_results_df, new_race_results_df],axis = 0)
            new_race_results_df["course_len"] = new_race_results_df["course_len"].apply(_to_int)
            # csv/pickleに書き込む
            save_race_results_dataset(place_id, day.year, new_race_results_df)
        except Exception as e:
            race_results_error(e)

def make_yearly_race_results_dataset(place_id, year = date.today().year):
    """ 開催コースと年を指定して、1年間のrace_resultsデータセットを作成 
        Args:
            place_id (int) : 開催コースid
            year(int) : 年（初期値：今年）
    """
    #race_id_listの取得
    race_id_list = get_race_id.get_year_id_all(place_id, year)

    # スクレイピング
    race_results_df = scrape_race_results_dataframe(race_id_list)
    race_results_df["course_len"] = race_results_df["course_len"].apply(_to_int)

    # csv/pickleに書き込む
    save_race_results_dataset(place_id, year, race_results_df)

def make_up_to_day_dataset(place_id, day = date.today()):
    """ 指定日までの、年間のrace_resultsデータセットを作成 
        Args:
            place_id (int) : 開催コースid
            day(int) : 日（初期値：今日）
    """
    #race_id_listの取得
    race_id_list = get_race_id.get_past_year_id(place_id, day)

    # スクレイピング
    race_results_df = scrape_race_results_dataframe(race_id_list)
    race_results_df["course_len"] = race_results_df["course_len"].apply(_to_int)

    try:
        # csv/pickleに書き込む
        save_race_results_dataset(place_id, day.year, race_results_df)
    except Exception as e:
            race_results_error(e) 

def weekly_update_race_results(day = date.today()):
    """ 指定した日にちから、１週間分のデータセットを更新  
    Args:
        day(Date) : 日（初期値：今日）
    """ 
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        print("[WeeklyUpdate]" + name_header.PLACE_LIST[place_id -1] + " RaceResults")
        update_race_results_dataset(place_id, day)
        export_race_info_per_race(place_id, day.year)
        split_race_results_by_year(place_id, day.year)

def montly_update_race_results(day = date.today()):
    """ 指定した日にちまでのその年のデータセットを更新  
    Args:
        day(Date) : 日（初期値：今日）
    """ 
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        print("[MonthlyUpdate]" + name_header.PLACE_LIST[place_id -1] + " RaceResults")
        make_up_to_day_dataset(place_id, day)

def make_all_race_results(year = date.today().year):
    """ 指定した年までの、すべてのデータセットを作成 
    Args:
        day(Date) : 日（初期値：今日）
    """ 
    for y in range(2019, year + 1):
        for place_id in range(1, len(name_header.PLACE_LIST) + 1):
            print("[NewMake]" + str(y) + ":" + name_header.PLACE_LIST[place_id -1] + " RaceResults")
            make_yearly_race_results_dataset(place_id, y)

def split_race_results_by_year(place_id, year):
    """ place_idのrace_resultsを指定年で分割して取得
        Args:
            place_id(int) : place_id
            year(int) : 年 
    """
    # CSVファイルの読み込み（ファイル名は適宜変更してね）
    df = get_race_results_csv(place_id, year)

    # 出力先ディレクトリ（必要なら作成）
    output_dir = name_header.DATA_PATH + "RaceResults\\" + name_header.PLACE_LIST[place_id - 1] +  '//' + str(year)
    os.makedirs(output_dir, exist_ok=True)

    # race_id ごとに分割して保存
    for race_id, group in df.groupby(df.index):
        try:
            output_path = os.path.join(output_dir, f"{race_id}.csv")
            # 重複している内容を消去
            group = group.drop_duplicates(subset="馬名", keep="first")
            group.to_csv(output_path, index=True)
            # print(f"保存完了: {output_path}")
        except Exception :
            print("レース結果の分割に失敗しました。", race_id)
    print(year, name_header.PLACE_LIST[place_id - 1], "CompleteResultsSplit")

if __name__ == "__main__":
    # montly_update_race_results()
    # weekly_update_race_results()
    make_all_race_results()