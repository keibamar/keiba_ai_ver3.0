import os
import sys
import re

from datetime import date
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import name_header

# # --- 固定リスト ---
CLASSES = ["all", "未勝利","新馬", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"]
GROUNDS = ["全", "良", "稍重", "重", "不良"]

def _parse_passages(pass_str):
    """ 通過文字列から整数のリストを返す（例 "1-1-1-1" -> [1,1,1,1]）。
        空・NaN・不正は空リストを返す。 """
    if not isinstance(pass_str, str) or pass_str.strip() == "":
        return []
    # 数字を抽出（通過はハイフン区切りのケースが多い）
    nums = re.findall(r'\d+', pass_str)
    return [int(n) for n in nums] if nums else []

def analyze_winners_multi_years(place_id, start_year = 2019, year = date.today().year):
    """
    各年度（start_year〜今年）について analyze_winners_with_passages() を実行し、
    年ごとの結果 + 全期間平均の DataFrame を返す。

    Parameters
    ----------
    base_dir : str
        CSV ファイルのディレクトリパス（例: "./data/"）
        → ファイル名は "{year}.csv" のようにする前提。
    place_id : int
        競馬場ID (1〜10)
    start_year : int
        集計を始める年（例: 2020）

    Returns
    -------
    results_by_year : dict[int, pd.DataFrame]
        年度別の結果を格納した辞書。
    total_df : pd.DataFrame
        全期間の平均結果。
    """

    results_by_year = {}
    base_dir = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[int(place_id) -1] + "//"
    for year in range(start_year, year + 1):
        csv_path = os.path.join(base_dir, f"{year}_race_results.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ {csv_path} が見つかりません。スキップします。")
            continue
        # print(f"📘 {year}年のデータを処理中 ...")
        df_year = analyze_winners(place_id, year)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year

    # --- 全年度をまとめる ---
    if not results_by_year:
        print("❌ 有効なデータがありません。")
        return {}, pd.DataFrame()

    combined_df = pd.concat(results_by_year.values(), ignore_index=True)

    # --- 全期間平均を算出（年列を無視） ---
    numeric_cols = ["上り", "通過1", "通過2", "通過3", "通過4"]
    group_cols = ["race_type", "course_len", "ground_state", "class"]

    total_df = (
        combined_df.groupby(group_cols, dropna=False)[numeric_cols]
        .mean()
        .round(1)
        .reset_index()
    )

    # 並び順と列順を統一（既存関数と同じ）
    total_df["race_type"] = pd.Categorical(total_df["race_type"], categories=["芝", "ダート"], ordered=True)
    total_df["class"] = pd.Categorical(total_df["class"], categories=CLASSES, ordered=True)
    total_df["ground_state"] = pd.Categorical(total_df["ground_state"], categories=GROUNDS, ordered=True)
    total_df = total_df.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)
    total_df = total_df.reindex(columns=["race_type", "course_len", "ground_state", "class", "上り", "通過1", "通過2", "通過3", "通過4"])

    print(f"✅ {name_header.PLACE_LIST[int(place_id) - 1]} 上り/通過全期間平均（{start_year}〜{year}）を作成しました。")
    return total_df

def analyze_winners(place_id, year):
    """
    place_id のコース定義(name_header.COURSE_LISTS[place_id-1])に従って
    勝ち馬の「上り」と「通過1..4」を算出する。
    """
    # --- 生データ読み込み（race_id を列に） ---
    csv_path = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[int(place_id) -1] + "//" + f"{year}_race_results.csv"
    if os.path.isfile(csv_path):
        df_raw = pd.read_csv(csv_path, dtype=str, index_col=0).reset_index().rename(columns={"index": "race_id"})
    else:
        return pd.DataFrame()

    # --- race_idごとの出走頭数（除外: '除' などを除く）を取得 ---
    # 着順カラムが文字列のままなので '除' を直接判定できる
    runners_count = df_raw.groupby("race_id").apply(lambda g: g[g["着順"] != "除"].shape[0]).to_dict()

    # --- コピーして数値変換（その後の集計に使う） ---
    df = df_raw.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["上り"] = pd.to_numeric(df["上り"], errors="coerce")
    df["course_len"] = pd.to_numeric(df["course_len"], errors="coerce")

    # 勝ち馬のみ
    winners = df[df["着順"] == 1].copy()
    if winners.empty:
        return pd.DataFrame()

    all_results = []
    courses = name_header.COURSE_LISTS[place_id - 1]

    for race_type, course_len in courses:
        # ベース（その条件で勝ち馬の行を抽出）
        base_data = winners[
            (winners["race_type"] == race_type) &
            (winners["course_len"] == float(course_len))
        ]

        # 条件ごとに集計
        for cls in CLASSES:
            for grd in GROUNDS:
                tmp = base_data.copy()
                if cls != "all":
                    tmp = tmp[tmp["class"] == cls]
                if grd != "全":
                    tmp = tmp[tmp["ground_state"] == grd]

                # 上り平均
                avg_last = tmp["上り"].mean() if not tmp.empty else None

                # --- 通過ステージごとの平均を集計（最大4ステージを想定） ---
                # 各ステージの値を溜めるリスト
                stage_vals = {1: [], 2: [], 3: [], 4: []}

                for _, winner_row in tmp.iterrows():
                    race_id = str(winner_row["race_id"])
                    num_runners = runners_count.get(int(race_id), None)
                    pass_str = df_raw.loc[df_raw["race_id"] == int(race_id), "通過"]
                    # 通常は 1 行だけだが念のため最初の非NA文字列を採用
                    pass_val = ""
                    if not pass_str.empty:
                        # pass_str は Series。取れる最初の非NAを探す
                        for v in pass_str.values:
                            if isinstance(v, str) and v.strip() != "":
                                pass_val = v
                                break
                    passages = _parse_passages(pass_val)
                    if not passages or num_runners is None or num_runners == 0:
                        continue
                    # 各ステージについて、18頭換算で正規化して格納
                    for idx, pos in enumerate(passages[:4], start=1):
                        # pos は 1-origin の着順
                        normalized = (pos / num_runners) * 18.0
                        stage_vals[idx].append(normalized)

                # 平均化（存在しなければ None）
                avg_stage = {}
                for i in range(1,5):
                    if stage_vals[i]:
                        avg_stage[i] = round(float(np.mean(stage_vals[i])), 2)
                    else:
                        avg_stage[i] = None

                all_results.append({
                    "race_type": race_type,
                    "course_len": int(course_len),
                    "ground_state": grd,
                    "class": cls,
                    "上り": round(avg_last, 2) if avg_last is not None else None,
                    "通過1": avg_stage[1],
                    "通過2": avg_stage[2],
                    "通過3": avg_stage[3],
                    "通過4": avg_stage[4],
                })

    # DataFrame化
    df_result = pd.DataFrame(all_results).round(1)
    # df_result = (.round(1))

    # 並び順を固定（芝→ダート, 距離昇順, class→ground_state）
    df_result["race_type"] = pd.Categorical(df_result["race_type"], categories=["芝", "ダート"], ordered=True)
    df_result["class"] = pd.Categorical(df_result["class"], categories=CLASSES, ordered=True)
    df_result["ground_state"] = pd.Categorical(df_result["ground_state"], categories=GROUNDS, ordered=True)
    df_result = df_result.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)

    # 列順を ground_state → class に入れ替えたい場合はここで reorder（あなたの希望に合わせる）
    df_result = df_result.reindex(columns=["race_type", "course_len", "ground_state", "class", "上り", "通過1", "通過2", "通過3", "通過4"])

    return df_result

def winners_time_update(place_id, year):
    # --- 使用例 ---
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        # 各年の記録を計算
        for year in range(2019,date.today().year + 1):
            result = analyze_winners(place_id, year)
            if not result.empty:
                output_path = name_header.DATA_PATH + "//AverageTimes//" + name_header.PLACE_LIST[int(place_id) -1] + "//" + f"{year}_wineer_time.csv"
                result.to_csv(output_path)
        # totalの記録を計算
        total_df = analyze_winners_multi_years(place_id, 2019, year)
        total_ouutput_path = name_header.DATA_PATH + "//AverageTimes//" + name_header.PLACE_LIST[int(place_id) -1] + "//" + "total_wineer_time.csv"
        total_df.to_csv(total_ouutput_path)

def weekly_winners_time_update(year = date.today().year):
    # --- 使用例 ---
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        winners_time_update(place_id, year)

if __name__ == '__main__':
    # --- 使用例 ---
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        # 各年の記録を計算
        for year in range(2019,date.today().year + 1):
            result = analyze_winners(place_id, year)
            if not result.empty:
                output_path = name_header.DATA_PATH + "//AverageTimes//" + name_header.PLACE_LIST[int(place_id) -1] + "//" + f"{year}_wineer_time.csv"
                result.to_csv(output_path)
        # totalの記録を計算
        total_df = analyze_winners_multi_years(place_id, 2019, year)
        total_ouutput_path = name_header.DATA_PATH + "//AverageTimes//" + name_header.PLACE_LIST[int(place_id) -1] + "//" + "total_wineer_time.csv"
        total_df.to_csv(total_ouutput_path)

