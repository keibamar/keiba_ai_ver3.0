import os
import re
import sys

from datetime import date, timedelta
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# pycache を生成しない
sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import get_race_id
import name_header
import horse_peds
import race_results
import scraping
import past_performance

def peds_dataset_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def make_peds_dataset_from_race_results(place_id, year):
    """ peds_datasetをrace_resultsから作成する 
        Args:
            place_id (int) : 開催コースid
            year(int) : 開催年
    """
    #レース結果のcsv情報を取得
    df_course = race_results.get_race_results_csv(place_id, year)
    
    if not df_course.empty:
        # ホースIDの抽出
        horse_id_list = df_course["horse_id"].to_list()

        # 血統データの取得
        df_peds = get_peds_dataset_from_horse_id_list(horse_id_list)
        
        # datasetの保存
        save_peds_dataset(df_peds, place_id, year)

def save_peds_dataset(peds_df, place_id, year):
    """ peds_datasetのDataFrameを保存 
        Args:
            peds_df(pd.DataFrame)
            horse_id (int) :horse_id
            past_performance_df.DataFrame） : past_performanceのデータセット
    """
    try:
        if any(peds_df):
            # 重複している内容を消去
            peds_df = peds_df.drop_duplicates(keep = 'first')
            # str型に変更
            peds_df = peds_df.astype(str)
            # ローカル保存
            peds_df.to_csv(name_header.DATA_PATH + "/PedsResults/" +  name_header.PLACE_LIST[place_id - 1] + '//' + str(year) + '_peds.csv')
            peds_df.to_pickle(name_header.DATA_PATH + "/PedsResults/" + name_header.PLACE_LIST[place_id - 1] + '//' + str(year) + '_peds.pickle')
    except Exception as e:
            peds_dataset_error(e)

def get_peds_dataset_csv(place_id, year):
    """ pedsのcsvを取得する 
        Args:
            place_id (int) : 開催コースid
            year(int) : 開催年
    """
    # csvを読み込む 
    path = name_header.DATA_PATH + "/PedsResults/" +  name_header.PLACE_LIST[place_id - 1] + '//' + str(year) + '_peds.csv'
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col = 0, dtype = str)
        df.fillna(np.nan)  
    else :
        df = pd.DataFrame()
    return df

def get_peds_data_dataset_csv(place_id, year):
    """ peds_dataのcsvを取得する 
        Args:
            place_id (int) : 開催コースid
            year(int) : 開催年
    """
    # csvを読み込む 
    path = name_header.DATA_PATH + "/PedsResults/" +  name_header.PLACE_LIST[place_id - 1] + '//' + str(year) + '_peds_data.csv'
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col = 0, dtype = str)
        df.fillna(np.nan)  
    else :
        df = pd.DataFrame()
    return df

def get_peds_dataset_from_horse_id_list(horse_id_list):
    """ horse_id_listからpeds_datasetのDatasetを作成 
        Args:
            horse_id_list : horse_idのリスト
        Returns:
            peds_df(pd.DataFrame) : peds_dataset
    """
    # 血統データの取得
    peds_df = pd.DataFrame()
    for horse_id in horse_id_list:
        peds_df = pd.concat([peds_df,horse_peds.get_horse_peds_csv(horse_id).T],axis = 0)
    return peds_df

def merge_pedsdata_with_race_results(place_id, year):
    """ 血統とレース結果の統合データ作成 
        Args:
            place_id (int) : 開催コースid
            year(int) : 開催年
    """
    #レース結果のcsv情報を取得
    df_course = race_results.get_race_results_csv(place_id, year)
    if not df_course.empty:
        # レース結果から，"着順","race_type","course_len","ground_state","class"を抽出
        df_result = df_course['着順']
        df_result =pd.concat([df_result,df_course['race_type']],axis = 1)
        df_result =pd.concat([df_result,df_course['course_len']],axis = 1)
        df_result =pd.concat([df_result,df_course['ground_state']],axis = 1)
        df_result =pd.concat([df_result,df_course['class']],axis = 1)

        # 血統データの取得
        df_peds = get_peds_dataset_csv(place_id, year)    
        # horse_idに一致する血統情報を抽出
        df_peds_result =pd.DataFrame()
        for id in range(len(df_result.index)):
            num = df_course.iloc[id,:]
            num = int(num['horse_id'])
            if num in df_peds.index:
                df_peds_result = pd.concat([df_peds_result,df_peds.loc[num,:]], axis=1)
            else :
                horse_peds_data = horse_peds.get_horse_peds_csv(str(num))
                if horse_peds_data.empty:
                    horse_peds_data = pd.DataFrame( np.nan, index=df_peds_result.index, columns = [num])
                df_peds_result = pd.concat([df_peds_result,horse_peds_data], axis=1)
       
        df_peds_result = df_peds_result.reset_index(drop = True).T

        # index/columnsの整理
        df_peds_result = df_peds_result.reset_index(drop = True)
        df_peds_result.columns = [f'peds_{i}' for i in range(len(df_peds_result.columns))]
        # 血統情報を結合
        df_result = pd.concat([df_result.reset_index(drop = True), df_peds_result], axis = 1)
        df_result.index = df_course.index
        df_result.index.name = "index"
        
        # データセットの保存
        df_result.to_csv(name_header.DATA_PATH + "/PedsResults/" +  name_header.PLACE_LIST[place_id - 1] + '//' + str(year) + '_peds_data.csv')
        df_result.to_pickle(name_header.DATA_PATH + "/PedsResults/" +  name_header.PLACE_LIST[place_id - 1] + '//' + str(year) + '_peds_data.pickle')

def calc_peds_placed_rate(peds_data):
    """データセットから着度数を計算
        Args: 
            peds_data(pd.DatatFrame): 血統毎のレース結果データセット
        Returns:
            placed_rate(pd.DataFrame):着度数
    """
    placed_rate = [peds_data[peds_data['着順'] == '1']['着順'].count(), 
                        peds_data[peds_data['着順'] == '2']['着順'].count(),
                        peds_data[peds_data['着順'] == '3']['着順'].count(),
                        len(peds_data) - (peds_data[peds_data['着順'] == '1']['着順'].count() + peds_data[peds_data['着順'] == '2']['着順'].count() + peds_data[peds_data['着順'] == '3']['着順'].count())]
    return placed_rate

def calc_peds_data(df_result, course_len, race_class):
    """血統の着度数を計算
        Args: 
            df_result(pd.DatatFrame): 血統毎のレース結果データセット
            course_len(str) : コースキョリ
            race_class(sre) : クラス
        Returns:
            return_df(pd.DataFrame) : 血統着度数データセット
    """
    return_data = []
    # 同競馬場スコア
    return_data.append(calc_peds_placed_rate(df_result))

    # 同コース
    df_result = df_result[df_result['course_len'] == str(course_len)]
    return_data.append(calc_peds_placed_rate(df_result))

    # 同条件
    df_result = df_result[df_result['class'] == str(race_class)]
    return_data.append(calc_peds_placed_rate(df_result))

    # リストのデータフレーム化
    return_df = pd.DataFrame(zip(*return_data), columns = ["place", "course", "class"])

    return return_df

def get_race_type_data(df_result, race_type, ground_state):
    """df_resultのレースタイプと馬場状態を抽出
        Args:
            df_result(pd.DataFrame) : race_resultデータセット
            race_type(str) : コースタイプ
            ground_state(str) : 馬場状態
        Returns:
            df_result(pd.DataFrame) : race_result
    """
    df_result = df_result[df_result['race_type'] == race_type]
    df_result = df_result[df_result['ground_state'] == ground_state]

    return df_result

def peds_index(father, mother_father, course_info, year):
    """血統情報の抽出
        Args: 
            father(str) : 父
            mother_father(str) : 母父
            course_info : place_id, race_type, course_len, ground_state, race_class
            year(int) : 年
        Returns:
            return_df(pd.DataFrame) : 血統の着度数データ
    """
    place_id = course_info[0]
    race_type = course_info[1]
    course_len = course_info[2]
    ground_state = course_info[3]
    race_class = course_info[4]

    df_peds = pd.DataFrame()
    # 各年度のデータを抽出
    for y in range(2019, int(year)):
        df_peds = pd.concat([df_peds, get_peds_data_dataset_csv(place_id, y)])
    df_peds = df_peds.reset_index(drop = True)
    
    # 着度数を格納
    return_df = pd.DataFrame()
    
    # 父情報を習得
    peds_data_father = df_peds[df_peds['peds_0'] == father]
    peds_data_father =  get_race_type_data(peds_data_father, race_type, ground_state)
    return_df = pd.concat([return_df,calc_peds_data(peds_data_father, course_len, race_class)],axis = 0)

    # 母情報を習得
    peds_data_mother_father = df_peds[df_peds['peds_4'] == mother_father]
    peds_data_mother_father =  get_race_type_data(peds_data_mother_father, race_type, ground_state)
    return_df = pd.concat([return_df,calc_peds_data(peds_data_mother_father, course_len, race_class)],axis = 0)
    
    # 父×母父情報を習得
    peds_data = df_peds[df_peds['peds_0'] == father]
    peds_data = peds_data[peds_data['peds_4'] == mother_father]
    peds_data = get_race_type_data(peds_data, race_type, ground_state)
    return_df = pd.concat([return_df,calc_peds_data(peds_data, course_len, race_class)],axis = 0)

    return return_df

def output_results(df_peds_results):
    """血統ごとの着順集計してCSV出力"""
    if df_peds_results.empty:
        return

    result_list = []
    for peds, sub_df in df_peds_results.groupby("peds_0"):
        first = (sub_df["着順"] == 1).sum()
        second = (sub_df["着順"] == 2).sum()
        third = (sub_df["着順"] == 3).sum()
        others = ((sub_df["着順"] > 3) & (sub_df["着順"].notna())).sum()

        result_list.append({
            "血統": peds,
            "1着": first,
            "2着": second,
            "3着": third,
            "着外": others
        })

    result_df = pd.DataFrame(result_list)
    result_df = result_df.sort_values(by=["1着", "2着", "3着"], ascending=False)

    return result_df

def aggregate_peds_results(place_id, year):
    """
    各コース（芝/ダート・距離）× 馬場状態 × クラスごとに血統成績を集計。
    全馬場状態(all)、全クラス(all)の集計も同時に出力。
    """
    # === データ読み込み ===
    df = get_peds_data_dataset_csv(place_id, year)
    if df.empty:
        print("not PedsResultData:", name_header.PLACE_LIST[place_id - 1], year)
        return
    
    output_dir = os.path.join(name_header.DATA_PATH, "PedsResults", name_header.PLACE_LIST[place_id - 1], str(year))
    os.makedirs(output_dir, exist_ok=True)

    # 着順を数値化
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")

    # === race_type, course_len ごとに処理 ===
    for (race_type, course_len), df_group in df.groupby(["race_type", "course_len"]):
        ground_list = sorted(df_group["ground_state"].dropna().unique())

        # === 各馬場状態ごと ===
        for ground_state in ground_list + ["all"]:
            if ground_state == "all":
                df_ground = df_group
            else:
                df_ground = df_group[df_group["ground_state"] == ground_state]

            if df_ground.empty:
                continue
            
            # === 全クラスまとめ(all) ===
            result_all_classes = []
            result_all = output_results(df_ground)
            if not result_all.empty:
                result_all.insert(0, "クラス", "all")
                result_all_classes.append(result_all)

            # === クラス別処理 ===
            for class_name, df_class in df_ground.groupby("class"):
                result_df = output_results(df_class)
                if not result_df.empty:
                    result_df.insert(0, "クラス", class_name)
                    result_all_classes.append(result_df)

            # === 全クラス結合 ===
            if result_all_classes:
                final_df = pd.concat(result_all_classes, ignore_index=True)
            else:
                continue

            # === 出力 ===
            file_name = f"{race_type}_{course_len}m_{ground_state}.csv"
            output_path = os.path.join(output_dir, file_name)
            final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"make peds_result {name_header.PLACE_LIST[place_id -1]} {year} {race_type} {course_len}m")


def aggregate_total_peds_results(place_id, start_year=2019, end_year=date.today().year):
    """
    各年度のPeds結果CSVを統合し、Totalディレクトリに合計結果を出力。

    Parameters
    ----------
    place_id : int
        競馬場ID
    start_year : int
        集計開始年
    end_year : int
        集計終了年（含む）
    """

    base_dir = os.path.join(name_header.DATA_PATH, "PedsResults", name_header.PLACE_LIST[place_id - 1])
    output_dir = os.path.join(base_dir, "Total")
    os.makedirs(output_dir, exist_ok=True)

    # === 全ファイル名の集合を取得 ===
    all_files = set()
    for year in range(start_year, end_year + 1):
        year_dir = os.path.join(base_dir, str(year))
        if not os.path.exists(year_dir):
            continue
        csv_files = [os.path.basename(p) for p in glob(os.path.join(year_dir, "*.csv"))]
        all_files.update(csv_files)

    print(f"🔍 集計対象ファイル数: {len(all_files)} ({start_year}–{end_year})")

    # === 各ファイル名ごとに集計 ===
    for file_name in sorted(all_files):
        merged_df_list = []

        for year in range(start_year, end_year + 1):
            csv_path = os.path.join(base_dir, str(year), file_name)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if not df.empty:
                    df["year"] = year
                    merged_df_list.append(df)

        if not merged_df_list:
            continue

        df_all = pd.concat(merged_df_list, ignore_index=True)

        # === 集計 ===
        agg_df = (
            df_all.groupby(["クラス", "血統"], as_index=False)[["1着", "2着", "3着", "着外"]]
            .sum()
        )
        # === クラスの表示順を定義 ===
        CLASS_ORDER = ["all", "未勝利", "新馬", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"]

        # === クラス順序をカテゴリ型で保持 ===
        agg_df["クラス"] = pd.Categorical(agg_df["クラス"], categories=CLASS_ORDER, ordered=True)
        
        # === クラス → 着順 の順でソート ===
        agg_df = agg_df.sort_values(
            by=["クラス", "1着", "2着", "3着"],
            ascending=[True, False, False, False]
        ).reset_index(drop=True)

        # === 出力 ===
        output_path = os.path.join(output_dir, file_name)
        agg_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"✅ Total集計完了: {file_name}")

    print(f"\n🎯 すべてのTotal集計が完了しました -> {output_dir}")
        

def update_peds_dataset(place_id, day = date.today()):
    """ 指定したコースの指定日から、１週間分のデータセットを更新  
    Args:
        place_id (int) : 開催コースid
        day(Date) : 日（初期値：今日）
    """ 
    race_id_list = get_race_id.get_past_weekly_id(place_id, day)
    horse_id_list = past_performance.get_horse_id_list_from_race_id_list(race_id_list)
    if any(horse_id_list):
        # 新しいデータセットの取得
        new_peds_df = get_peds_dataset_from_horse_id_list(horse_id_list)
        # 既存のデータセットを取得
        old_peds_df = get_peds_dataset_csv(place_id, day.year)
        # 新しいデータセットを統合
        new_peds_df = pd.concat([old_peds_df,new_peds_df],axis = 0)
        save_peds_dataset(new_peds_df, place_id, day.year)
    
def weekly_update_pedsdata(day = date.today()):
    """ 指定した日にちから、１週間分のデータセットを更新  
    Args:
        day(Date) : 日（初期値：今日）
    """ 
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        print("[WeeklyUpdate]" + name_header.PLACE_LIST[place_id -1] + " RaceResults")
        update_peds_dataset(place_id, day)
        merge_pedsdata_with_race_results(place_id, day.year)
        # 集計結果を更新
        aggregate_peds_results(place_id, day.year)
        aggregate_total_peds_results(place_id = place_id, end_year = day.year)

def monthly_update_pedsdata(day = date.today()):
    """ 指定した日にちまでのその年のデータセットを更新  
    Args:
        day(Date) : 日（初期値：今日）
    """ 
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        print("[MonthlyUpdate]" + name_header.PLACE_LIST[place_id -1] + " PedsResults")
        make_peds_dataset_from_race_results(place_id, day.year)
        merge_pedsdata_with_race_results(place_id, day.year)
        # 集計結果を更新
        aggregate_peds_results(place_id, day.year)
        aggregate_total_peds_results(place_id = place_id, end_year = day.year)
def make_all_pedsdata(year = date.today().year):
    """ 指定した年までの、すべてのデータセットを作成 
    Args:
        day(Date) : 日（初期値：今日）
    """ 
    for y in range(2019, year + 1):
        for place_id in range(1, len(name_header.PLACE_LIST) + 1):
            print("[NewMake]" + str(y) + ":" + name_header.PLACE_LIST[place_id -1] + " PedsResults")
            make_peds_dataset_from_race_results(place_id, y)
            merge_pedsdata_with_race_results(place_id, y)
            # 集計結果を更新
            aggregate_peds_results(place_id, y)
            aggregate_total_peds_results(place_id = place_id, end_year = y)

def make_peds_results(year = date.today().year):
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        aggregate_peds_results(place_id, year)

if __name__ == "__main__":
#    make_peds_dataset_from_race_results(3, 2024)
#    make_peds_dataset_from_race_results(5, 2024)
#    make_peds_dataset_from_race_results(8, 2024)
   for place_id in range(1, len(name_header.PLACE_LIST) + 1):
       update_peds_dataset(place_id)
#    make_all_pedsdata()