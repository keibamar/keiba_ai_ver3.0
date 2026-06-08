import os
import re
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# pycache を生成しない
sys.dont_write_bytecode = True

# web/src を import パスに追加（config パッケージを解決するため）
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # web/src
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# プロジェクトルートを正しく計算して libs を追加（libs はプロジェクトルート直下）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # project root (keiba_ai_ver2.0)
LIBS_PATH = os.path.join(PROJECT_ROOT, "libs")
if LIBS_PATH not in sys.path:
    sys.path.insert(0, LIBS_PATH)

from config.path import HORSE_ID_MAP_PATH, PAST_PERF_PATH, HORSE_PEDS_PATH, PEDS_RESULTS_PATH, TIME_INFO_PATH, RACE_CALENDAR_FOLDER_PATH
import name_header
import race_pages

try:
    from config.settings import RANK_COLORS, WAKU_COLORS
except Exception:
    # templates.py の定義名が異なる／未定義の場合のフォールバック
    from config import templates as templates_mod
    RANK_COLORS = getattr(templates_mod, "RANK_COLORS", {})
    WAKU_COLORS = getattr(templates_mod, "WAKU_COLORS", {})

# ---- ユーティリティ ----

def safe_read_csv(path: str, dtype=None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype, index_col=None)
    except Exception:
        return pd.read_csv(path, dtype=str, index_col=None)

def time_str_to_ms(t: str) -> Optional[int]:
    """
    '0:1:08.9', '1:32.7', '59.6' 等をミリ秒に変換。
    戻り値はミリ秒（int）。解析できない場合は None。
    """
    if pd.isna(t) or t is None:
        return None
    s = str(t).strip()
    if s == "" or s.lower() in ["nan", "---"]:
        return None
    # 形式を分解
    try:
        # 例: '0:1:08.9' -> ['0','1','08.9']  or '1:32.7' -> ['1','32.7']
        parts = s.split(":")
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = float(parts[2])
            total_ms = ((h * 60 + m) * 60 + sec) * 1000
        elif len(parts) == 2:
            m = int(parts[0])
            sec = float(parts[1])
            total_ms = (m * 60 + sec) * 1000
        else:
            # 秒のみ
            sec = float(parts[0])
            total_ms = sec * 1000
        return int(round(total_ms))
    except Exception:
        # 失敗したら None
        return None

def ms_to_time_str(ms: Optional[int]) -> str:
    """ミリ秒 -> 'M:SS.s' 表示（msがNoneは '-'）"""
    if ms is None:
        return "-"
    sec = ms / 1000.0
    m = int(sec // 60)
    s = sec - m * 60
    # 小数第一位に丸めて表示 (例 1:32.7)
    return f"{m}:{s:04.1f}"

def normalize_passage(pass_str: str, heads: Optional[int], target_heads=18) -> List[Optional[float]]:
    """
    通過文字列を数値化して正規化（target_headsスケール）しリスト返却。
    例: '6-5-4-4' -> [6_norm,5_norm,4_norm,4_norm]
    heads（出走頭数）が無ければ None を入れる。
    """
    if pd.isna(pass_str) or pass_str is None or str(pass_str).strip() == "":
        return []
    parts = [p.strip() for p in str(pass_str).split("-") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            pos = int(p)
            if heads and not pd.isna(heads) and float(heads) > 0:
                norm = pos * (target_heads / float(heads))
            else:
                norm = None
            out.append(norm)
        except:
            out.append(None)
    return out

# ---- データ読み込み関数 ----

def load_horse_id_map(path: str = HORSE_ID_MAP_PATH) -> pd.DataFrame:
    df = safe_read_csv(path, dtype=str)
    # 想定: columns = ['horse_id','horse_name'] か ['index','horse_name'] 等いろいろある可能性あり
    # 正規化して 'horse_id','horse_name' カラムを返す
    if df.empty:
        return df
    # もし index が 馬ID で 馬名が 0 列にある場合
    if "horse_id" not in df.columns and "馬名" not in df.columns:
        cols = df.columns.tolist()
        if len(cols) >= 2:
            # try horse_id in col0, horse_name in col1
            df = df.rename(columns={cols[0]: "horse_id", cols[1]: "馬名"})
        else:
            # fallback: treat index as horse_id and first column as name
            df = df.reset_index().rename(columns={"index": "horse_id", df.columns[0]: "馬名"})
    # strip
    df["馬名"] = df["馬名"].astype(str).str.strip()
    df["馬名"] = df["馬名"].astype(str).str.strip()
    return df[["horse_id", "馬名"]]

def get_avg_time(course_name, race_type, class_name, course_len, ground_state):
    """
    開催場名と条件から平均タイムを取得する関数

    Args:
        course_name (str): 開催場名（例: "中京"）
        race_type (str): レースタイプ（例: "芝" or "ダート"）
        class_name (str): クラス名（例: "3勝クラス", "未勝利", "オープン"）
        course_len (int or str): 距離（例: 1800）
        ground_state (str): 馬場状態（例: "良", "稍重", "重", "不"）

    Returns:
        float or None: 該当条件の平均タイム（存在しない場合は np.nan）
    """

    # --- 開催場からplace_id取得 ---
    try:
        place_id = name_header.NAME_LIST.index(course_name) + 1
    except ValueError:
        print(f"❌ 不明な開催場名: {course_name}")
        return np.nan

    # --- ファイルパス生成 ---
    avg_time_path = os.path.join(TIME_INFO_PATH, name_header.PLACE_LIST[place_id - 1], "total_avg_time.csv")
    if not os.path.exists(avg_time_path):
        print(f"⚠️ 平均タイムファイルが見つかりません: {avg_time_path}")
        return np.nan
    try:
        df = pd.read_csv(avg_time_path, dtype=str)
    except Exception as e:
        print(f"⚠️ CSV読み込み失敗: {avg_time_path} ({e})")
        return np.nan

    # --- 型変換 ---
    df["course_len"] = df["course_len"].astype(str).str.strip()
    df["avg_time"] = pd.to_numeric(df["avg_time"], errors="coerce")
    # --- 値の調整 ---
    if ground_state == "不良":
        ground_state = "不"
    if ground_state == "稍":
        ground_state = "稍重"
    if race_type == "ダ":
        race_type = "ダート"

    # --- 条件フィルタ ---
    cond = (
        (df["race_type"] == str(race_type)) &
        (df["course_len"] == str(course_len)) &
        (df["ground_state"] == str(ground_state)) &
        (df["class"] == str(class_name))
    )
    sub = df[cond]

    if sub.empty or sub["avg_time"].isna().all():
        print(f"⚠️ 該当データなし: {course_name} {race_type} {class_name} {course_len} {ground_state}")
        return np.nan

    # --- 平均値を返す（複数一致時は平均） ---
    return float(sub["avg_time"].values[0])

def get_horse_id_by_name(horse_name: str, map_df: pd.DataFrame) -> Optional[str]:
    if map_df is None or map_df.empty:
        return None
    # 完全一致, 部分一致の順で検索
    sel = map_df[map_df["馬名"] == horse_name]
    if not sel.empty:
        return sel.iloc[0]["horse_id"]
    # 部分一致（前方一致）
    sel = map_df[map_df["馬名"].str.contains(horse_name, na=False)]
    if not sel.empty:
        return sel.iloc[0]["馬名"]
    return None

def load_horse_peds(horse_id: str) -> Dict[str, Any]:
    path = os.path.join(HORSE_PEDS_PATH, f"{horse_id}.csv")
    df = safe_read_csv(path, dtype=str)
    if df.empty:
        return {}
    # df 形式が (index,row) になっているサンプルを考慮
    try:
        d = {}
        for idx, val in df.values:
            d[str(idx).strip()] = str(val).strip()
        # または、縦に peds_0, peds_1 のようなカラムがある場合
        if not d:
            for col in df.columns:
                d[col] = df.iloc[0].get(col, "")
    except Exception:
        d = {}
    return d

def load_peds_results(place_id: int, race_type: str, course_len: int, ground_state: str) -> pd.DataFrame:
    """
    PedsResults/{place_name}/Total/{condition_file}.csv を返す
    condition_file は e.g. 'ダート_1400m_良.csv' のようなファイル名（拡張子なしでも可）。
    """
    fname = race_type + "_" + str(course_len) + "m_" + ground_state
    if not fname.lower().endswith(".csv"):
        fname = fname + ".csv"
    path = os.path.join(PEDS_RESULTS_PATH, name_header.PLACE_LIST[place_id - 1], "Total", fname)
    return safe_read_csv(path, dtype=str)

def load_past_performance(horse_id: str) -> pd.DataFrame:
    path = os.path.join(PAST_PERF_PATH, f"{horse_id}.csv")
    df = safe_read_csv(path, dtype=str)
    if df.empty:
        return df
    # 日付列があれば parse
    if "日付" in df.columns:
        try:
            df["日付_parsed"] = pd.to_datetime(df["日付"], errors="coerce", dayfirst=False)
        except Exception:
            df["日付_parsed"] = pd.to_datetime(df["日付"], errors="coerce")
    # normalize column names: remove leading/trailing spaces
    df.columns = [c.strip() for c in df.columns]
    return df

# ---- 解析関数 ----
def peds_results_for_bloodline(place_id: int, race_type: str, course_len: int, ground_state: str, peds0_name: str ) -> pd.DataFrame:
    """
    指定血統（peds0_name）の PedsResults が存在すれば dataframe を返す。
    ファイル内の 'クラス' 列ごとにフィルタして返す（呼び出し側で利用）。
    """
    df = load_peds_results(place_id, race_type, course_len, "all")
    if df.empty:
        return pd.DataFrame()
    # normalize columns
    df.columns = [c.strip() for c in df.columns]
    pattern = rf"\b{re.escape(peds0_name)}\b"

    # 血統カラムが '血統' になっている想定
    if "血統" in df.columns:
        res = df[df["血統"].astype(str).str.contains(pattern, na=False, regex=True)]
        return res.copy()
    # 列名の揺らぎがあれば try その他
    for col in df.columns:
        if "血統" in col:
            res = df[df[col].astype(str).str.contains(pattern, na=False, regex=True)]
            return res.copy()
    return pd.DataFrame()

# NaN 判定して "-" に置き換え
def safe_value(val):
    if val is None or val == "None":
        return "-"
    try:
        if isinstance(val, float) and math.isnan(val):
            return "-"
        if pd.isna(val):
            return "-"
    except Exception:
        pass
    return val

def recent_5_performances(horse_id: str, date_str:str) -> List[Dict[str, Any]]:
    """
    PastPerformance/{horse_id}.csv から直近5走を取得して整形して返す。
    各エントリに日付、レース名、コース（距離/種別）、馬場、タイム(ms)、着差(ms)、上り、通過(正規化) 等を含む。
    """
    df = load_past_performance(horse_id)
    if df.empty:
        return []

    # --- 日付の正規化 ---
    if "日付" in df.columns:
        df["日付_parsed"] = pd.to_datetime(df["日付"], errors="coerce")
    else:
        print(f"⚠️ '日付'列が見つかりません (horse_id={horse_id})")
        return []

    # --- race_day を datetime に変換 ---
    try:
        race_day_dt = datetime.strptime(str(date_str), "%Y%m%d")
    except ValueError:
        print(f"⚠️ race_dayの形式が不正です: {date_str}")
        return []

    # --- race_day より前のデータをフィルタ ---
    df = df[df["日付_parsed"] < race_day_dt]

    if df.empty:
        return []

    # --- ソート処理 ---
    if df["日付_parsed"].notna().any():
        df_sorted = df.sort_values("日付_parsed", ascending=False)
    else:
        df_sorted = df.iloc[::-1].copy()  # 日付がNaNなら逆順で想定

    # --- 最新5件などを抽出する場合 ---
    res = []
    count = 0
    for _, row in df_sorted.iterrows():
        if count >= 5:
            break
        count += 1
        # 基本フィールド
        race_id = row.get("race_id", "")
        date_raw = row.get("日付", "")
        waku = row.get("枠番", "")
        umaban = row.get("馬番", "")
        race_num = row.get("R", "")
        race_name = row.get("レース名", row.get("レース名", ""))
        pops = row.get("人気", "")
        try:
            pops = int(float(pops))
        except:
            pops = pops
        result = row.get("着順", "")
        race_type = row.get("race_type", "")
        course_len = row.get("course_len", "")
        class_name = row.get("class", "")
        course = str(course_len)
        ground = row.get("ground_state","")
        time_raw = row.get("タイム", "")
        t_ms = time_str_to_ms(time_raw)
        place_match = re.search(r"[0-9]*(東京|中山|阪神|京都|札幌|函館|福島|新潟|中京|小倉)[0-9]*", row.get("開催", ""))
        course_name = place_match.group(1) if place_match else ""
        diff_raw = row.get("着差", "")
        # 同条件の平均タイムを取得
        avg_time = get_avg_time(course_name, race_type, class_name, course_len, ground)
        if not avg_time is np.nan:
            try:
                diff_avg_ms = (t_ms - avg_time) / 1000
                diff_avg_ms = round(diff_avg_ms, 2)
            except Exception:
                diff_avg_ms = np.nan
        else:
            diff_avg_ms = np.nan
        try:
            time_raw = re.sub(r"^0:", "", time_raw)
        except Exception:
            time_raw = np.nan

        # 通過の正規化
        heads = None
        if "頭 数" in row:
            try:
                heads = int(str(row.get("頭 数")).strip())
            except:
                heads = None
        elif "頭数" in row:
            try:
                heads = int(str(row.get("頭数")).strip())
            except:
                heads = None
        passage = row.get("通過", row.get("通過", ""))
        passage_norm = normalize_passage(passage, heads)
        # レース名が取得できなかった場合、race_id_listから取得する
        if race_name is np.nan or None:
            # レース名・時刻取得
            race_date = datetime.strptime(date_raw, "%Y/%m/%d").strftime("%Y%m%d")
            race_info_path = os.path.join(RACE_CALENDAR_FOLDER_PATH, f"race_time_id_list/{race_date}.csv")
            if os.path.exists(race_info_path):
                df_info = pd.read_csv(race_info_path, dtype=str)
                match = df_info[df_info["race_id"].astype(str) == str(race_id)]
                if not match.empty:
                    race_name = str(match.iloc[0]["race_name"])

        res.append({
            "date": safe_value(date_raw),
            "date_parsed": safe_value(row.get("日付_parsed", None)),
            "race_name": safe_value(race_name),
            "race_num" : safe_value(race_num),
            "waku" : safe_value(waku),
            "umaban" : safe_value(umaban),
            "pops" : safe_value(pops),
            "result" : safe_value(result),
            "course_name" : safe_value(course_name),
            "course": safe_value(course),
            "race_type" : safe_value(race_type),
            "ground": safe_value(ground),
            "class_name": safe_value(class_name),
            "time_raw": safe_value(time_raw),
            "time_ms": safe_value(t_ms),
            "diff_avg_ms" : safe_value(diff_avg_ms),
            "diff_ms": safe_value(diff_raw),
            "上り": safe_value(row.get("上り", None)),
            "通過": safe_value(passage),
            "通過_norm": safe_value(passage_norm),
            "馬体重": safe_value(row.get("馬体重", None)),
            "枠": safe_value(row.get("枠 番", row.get("枠 番", row.get("枠", None)))),
            "馬番": safe_value(row.get("馬 番", row.get("馬 番", row.get("馬番", None)))),
            "人気": safe_value(row.get("人 気", row.get("人 気", row.get("人気", None)))),
            "着順": safe_value(row.get("着 順", row.get("着 順", row.get("着順", None)))),
            "斤量": safe_value(row.get("斤 量", row.get("斤 量", row.get("斤量", None)))),
        })
    return res

def turf_dirt_summary(horse_id: str, date_str: str) -> Dict[str, Any]:
    """
    PastPerformance から芝/ダートごとに:
      - 最速上り (msではなく表示は上りの値そのもの。該当レース情報付)
      - 平均上り
      - 平均通過位置（最後の通過位置を normalizedして平均）
    を計算して返す。
    """
    df = load_past_performance(horse_id)
    if df.empty:
        return {"芝": {}, "ダート": {}}

    # normalize column names
    df = df.copy()
    if "上り" in df.columns:
        df["上り_num"] = pd.to_numeric(df["上り"], errors="coerce")
    else:
        df["上り_num"] = pd.Series(dtype=float)
     # --- 日付の正規化 ---
    if "日付" in df.columns:
        df["日付_parsed"] = pd.to_datetime(df["日付"], errors="coerce")
    else:
        print(f"⚠️ '日付'列が見つかりません (horse_id={horse_id})")
        return []
    try:
        race_day_dt = datetime.strptime(str(date_str), "%Y%m%d")
    except ValueError:
        print(f"⚠️ race_dayの形式が不正です: {date_str}")
        return []
    # --- race_day より前のデータをフィルタ ---
    df = df[df["日付_parsed"] < race_day_dt]

    # 簡単に race surface 判定 : '芝' または 'ダート' を '馬 場' カラムで判定
    surface_summary = {}
    for surface in ["芝", "ダート"]:
        search_word = surface[0]
        sub = df[df.apply(lambda r: search_word in str(r.get("race_type", "")), axis=1)]
        if sub.empty:
            surface_summary[surface] = {"fastest_up": None, "fastest_up_info": None, "avg_up": None, "avg_pass_norm": None}
            continue
        # 最速上り
        if "上り_num" in sub.columns:
            s2 = sub.copy()
            s2 = s2[s2["上り_num"].notna()]
            if not s2.empty:
                idx = s2["上り_num"].idxmin()
                fastest_row = s2.loc[idx]
                fastest_up = fastest_row["上り_num"]
                place_match = re.search(r"[0-9]*(東京|中山|阪神|京都|札幌|函館|福島|新潟|中京|小倉)[0-9]*", fastest_row.get("開催", ""))
                course_name = place_match.group(1) if place_match else ""
                fastest_info = {
                    "date": fastest_row.get("日付", ""),
                    "race_name": fastest_row.get("レース名", ""),
                    "course_name": course_name,
                    "course_len": fastest_row.get("course_len", ""),
                    "馬場": fastest_row.get("ground_state", "")
                }
            else:
                fastest_up = None
                fastest_info = None
            avg_up = s2["上り_num"].mean() if not s2.empty else None
        else:
            fastest_up = None
            fastest_info = None
            avg_up = None

        # 平均通過（最終通過位置を見て normalize）
        norm_list = []
        for _, r in sub.iterrows():
            heads = None
            if "頭 数" in r and not pd.isna(r.get("頭 数")):
                try:
                    heads = int(r.get("頭 数"))
                except:
                    heads = None
            elif "頭数" in r and not pd.isna(r.get("頭数")):
                try:
                    heads = int(r.get("頭数"))
                except:
                    heads = None
            p = r.get("通過", "")
            arr = normalize_passage(p, heads)
            norm_list.append(arr)

        avg_pass_norm = calc_average_norm_passages(norm_list)

        # データの正規化
        avg_up = safe_value(avg_up)
        try:
            avg_up = round(float(avg_up), 2) 
        except:
            avg_up = "-"
        avg_pass_norm = safe_value(avg_pass_norm)

        surface_summary[surface] = {
            "fastest_up": fastest_up,
            "fastest_up_info": fastest_info,
            "avg_up": avg_up,
            "avg_pass_norm": avg_pass_norm,
            "count": len(sub)
        }
    return surface_summary

def calc_average_norm_passages(norm_list):
    """
    正規化済み通過位置のリスト（例: [[3.3, 4.5], [16.8, 16.8]]）を
    右詰めで整列し、各コーナーごとの平均を返す。
    """
    if not norm_list:
        return None

    # 最大長を求める
    max_len = max(len(x) for x in norm_list if isinstance(x, list))

    # 右詰め整列
    aligned = []
    for arr in norm_list:
        if not isinstance(arr, list) or len(arr) == 0:
            continue
        pad_len = max_len - len(arr)
        aligned.append([None] * pad_len + arr)

    # 平均計算（Noneを除外）
    avg_by_corner = []
    for i in range(max_len):
        vals = [row[i] for row in aligned if row[i] is not None]
        if vals:
            avg = round(sum(vals) / len(vals), 1)
            avg_by_corner.append(avg)
        else:
            avg_by_corner.append(None)

    return avg_by_corner


def same_course_best_time(
    horse_id: str,
    target_course_len: int,
    target_race_type: str,
    target_place_id: int,
    date_str : str
) -> Optional[Dict[str, Any]]:
    """
    PastPerformance から同じ開催場(place_id)、距離、馬場タイプ(芝/ダート)の持ち時計を返す。
    返値:
        {
            'time_ms': int,
            'time_str': '1:28.4',
            'date': '2024/10/13',
            'race_name': '2歳未勝利',
            'ground': '良',
            'place_id': '05_tokyo',
            'info_row': {...}
        }
    """
    df = load_past_performance(horse_id)
    if df.empty:
        return None

    df = df.fillna("").astype(str)
     # --- 日付の正規化 ---
    if "日付" in df.columns:
        df["日付_parsed"] = pd.to_datetime(df["日付"], errors="coerce")
    else:
        print(f"⚠️ '日付'列が見つかりません (horse_id={horse_id})")
        return []
    try:
        race_day_dt = datetime.strptime(str(date_str), "%Y%m%d")
    except ValueError:
        print(f"⚠️ race_dayの形式が不正です: {date_str}")
        return []
    # --- race_day より前のデータをフィルタ ---
    df = df[df["日付_parsed"] < race_day_dt]

    candidates = []

    for _, row in df.iterrows():
        _raw = str(row.get("開催", ""))
        # --- 開催場名を抽出 ---
        match = re.search(r"[0-9]*(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)[0-9]*", _raw)
        place_name = match.group(1) if match else ""

        if not place_name:
            continue

        # --- 開催場名 → place_id に変換 ---
        try:
            place_id = int(name_header.NAME_LIST.index(place_name)) + 1
        except ValueError:
            continue
       
        # --- 距離欄解析 ---
        dist_raw = str(row.get("race_type", "")).strip()
        if dist_raw.startswith("障"):
            continue
        race_type = "芝" if "芝" in dist_raw else "ダート" if "ダ" in dist_raw else ""

        dist = int(row.get("course_len", ""))
        if not race_type or not dist:
            continue

        # --- フィルタ条件 ---
        if place_id == target_place_id and race_type == target_race_type and dist == target_course_len:
            t_ms = time_str_to_ms(row.get("タイム", ""))
            if t_ms is not None:
                candidates.append((t_ms, row, place_id))

    if not candidates:
        return None

    # --- 最速タイムを選択 ---
    best_time_ms, best_row, place_id = sorted(candidates, key=lambda x: x[0])[0]

    return {
        "time_ms": best_time_ms,
        "time_str": ms_to_time_str(best_time_ms),
        "date": best_row.get("日付", ""),
        "race_name": best_row.get("レース名", ""),
        "ground": best_row.get("ground_state", best_row.get("ground_state", "")),
        "place_id": place_id,
        "info_row": best_row.to_dict() if hasattr(best_row, "to_dict") else {}
    }

# 血統名を抽出する
def extract_peds_name(peds0: str) -> str | None:
    if not peds0:
        return None
    
    # 前後の空白を削除
    peds0 = peds0.strip()
    
    # 先頭がカタカナならカタカナ部分だけ抽出
    match = re.match(r"^([\u30A0-\u30FFー]+)", peds0)
    if match:
        return match.group(1)
    
    # 先頭が英字なら英字部分だけ抽出
    match = re.match(r"^([A-Za-z\s]+)", peds0)
    if match:
        return match.group(1).strip()
    
    # どちらでもなければそのまま返す
    return peds0

# ---- 統合：馬ごとの全出力を作る関数 ----
def build_horse_report(horse_name: str, place_id: int, race_id: str, date_str: str) -> Dict[str, Any]:
    """
    horse_name から horse_id を特定し、①～④の情報を集めて dict で返す。
    - place_name: PedsResults の place フォルダ名 (e.g. '01_sapporo')
    - condition_file: PedsResults ファイル名 (例 'ダート_1400m_良.csv' または条件キー)
    - race_type, course_len: (任意) 現在見ているレースの種別・距離（同コース検索用）
    """
    # --- 基準レース情報取得 ---
    year = race_id[:4]
    race_type, course_len, ground_state, race_class = race_pages.get_race_info(year, place_id, race_id)
    if race_type == None and course_len == None and ground_state == None and race_class == None:
        return
    
    map_df = load_horse_id_map(HORSE_ID_MAP_PATH)
    hid = get_horse_id_by_name(horse_name, map_df)
    if not hid:
        return {"error": f"horse_id not found for {horse_name}"}

    # ① 血統(peds_0) と PedsResults（同コースの1,2,3,着外データ）
    peds = load_horse_peds(hid)
    peds0 = peds.get("peds_0") or peds.get("peds0") or peds.get("peds_0 ", None)
    peds0 = extract_peds_name(peds0)
    peds_results = None
    if peds0:
        peds_results = peds_results_for_bloodline(place_id, race_type, course_len, ground_state, peds0)
     # --- クラスでフィルタ ---
    filtered_df = peds_results[peds_results["クラス"].isin([race_class, "all"])].copy()
    if filtered_df.empty:
        print(f"⚠️ {peds0}: 該当コース({place_id}:{race_type}{course_len}) のデータがありません。")
        # return pd.DataFrame()
    else :
        # --- 勝率・複勝率を計算 ---
        for idx, row in filtered_df.iterrows():
            total = int(row["1着"]) + int(row["2着"]) + int(row["3着"]) + int(row["着外"])
            win_rate = (int(row["1着"]) / total) * 100 if total > 0 else 0.0
            fukusho_rate = ((int(row["1着"]) + int(row["2着"]) + int(row["3着"])) / total) * 100 if total > 0 else 0.0
            filtered_df.at[idx, "勝率"] = round(win_rate, 1)
            filtered_df.at[idx, "複勝率"] = round(fukusho_rate, 1)

        # --- 列の並びを指定（着外の横に勝率・複勝率を追加）---
        cols = ["クラス", "血統", "1着", "2着", "3着", "着外", "勝率", "複勝率"]
        filtered_df = filtered_df[cols]
    
    # ② 近5走
    recent5 = recent_5_performances(hid, date_str)

    # ③ 芝/ダート別サマリ
    surface_summary = turf_dirt_summary(hid, date_str)

    # ④ 同コースの持ち時計（もし race_type/course_len 与えられていれば）
    same_course_best = None
    if race_type and course_len:
        same_course_best = same_course_best_time(hid, course_len, race_type, place_id, date_str)

    # combine
    return {
        "horse_name": horse_name,
        "horse_id": hid,
        "peds0": peds0,
        "place_id": place_id,
        "race_type" : race_type, 
        "course_len" : course_len, 
        "ground_state": ground_state, 
        "race_class" :race_class,
        "peds_results": filtered_df if (isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty) else None,
        "recent5": recent5,
        "surface_summary": surface_summary,
        "same_course_best": same_course_best
    }

# ---- HTML 整形（簡易） ----
def get_time_diff_color(diff_str):
    """
    平均勝ち時計との差で色を返す
    - マイナスなら赤
    - 0.5秒以内なら オレンジ
    - それ以上なら 黒
    """
    try:
        diff_str = str(diff_str).strip()
        if not diff_str or diff_str == "-":
            return "black"
        
        diff_str_clean = diff_str.replace("秒", "").strip()
        diff_val = float(diff_str_clean)
        
        if diff_val < 0:
            return "red"        # マイナス = 赤（速い）
        elif 0 <= diff_val <= 0.2:
            return "orange"     # 0.2秒以内 = オレンジ（ほぼ同等）
        else:
            return "black"      # 0.2秒以上 = 黒（遅い）
    except:
        return "black"

def get_class_color(class_name):
    """クラスに基づいて背景色を返す"""
    class_colors = {
        "未勝利": "#fff0f0",    # 薄赤
        "新馬": "#ffe6e6",      # 少し濃い赤
        "1勝クラス": "#ffcccc", # さらに濃い赤
        "2勝クラス": "#ffb3b3", # もっと濃い赤
        "3勝クラス": "#ff9999", # さらに濃い赤
        "オープン": "#ff8080"   # 濃い赤
    }
    return class_colors.get(class_name, "#ffffff")

def get_race_type_color(race_type):
    """レースタイプに基づいて背景色を返す"""
    race_type_color = "#ffffff"
    if "ダ" in race_type:
        race_type_color = "#D2691E"
    elif "芝" in race_type:
        race_type_color = "#32CD32"
    elif "障" in race_type:
        race_type_color = "#8B4513"
    return race_type_color

def get_ground_state_color(ground_state):
    """馬場状態に基づいて背景色を返す"""
    ground_state_color = "#ffffff"
    if "稍" in ground_state:
        ground_state_color = "#e8e8e8"
    elif "重" in ground_state:
        ground_state_color = "#b0b0b0"
    elif "不" in ground_state:
        ground_state_color = "#808080"
    return ground_state_color

def horse_report_to_html(report: Dict[str, Any]) -> str:
    """
    build_horse_report の出力からスタイル付きHTML を作る。
    色付けロジック:
    - 着順、人気: 1着=黄色、2着=水色、3着=オレンジ
    - 平均時計との差: マイナス=赤、0.5秒以内=オレンジ、それ以上=黒
    - クラス: 未勝利→新馬→1勝→2勝→3勝→オープンで色が濃くなる
    - 枠番・馬番: 数字のみに色付け
    """
    if "error" in report:
        return f"<div class='horse-report error'>{report['error']}</div>"

    html = []
    html.append("<div class='horse-report' style='padding: 10px; background: #fafafa; border: 1px solid #ddd;'>")

    # PEDs
    p = report.get("peds0", {})
    if p:
        html.append(f"<h4>血統 (父) : <strong>{p}</strong></h4>")
    else:
        html.append("<h4>血統 (父) : - </h4>")
    
    race_type =  report.get("race_type", "-")
    course_len =  report.get("course_len", "-")
    ground_state =  report.get("ground_state", "-")
    place_id =  report.get("place_id", "-")
    place_num = name_header.NAME_LIST[place_id - 1]
    # PedsResults（summary）
    pr = report.get("peds_results")
    if pr is None or (isinstance(pr, pd.DataFrame) and pr.empty):
        html.append("<div>血統データなし</div>")
    else:
        html.append(f"<h4>🧬 {place_num} {race_type}{course_len}m ({ground_state})</h4>")
        html.append("<table style='width:100%; border-collapse: collapse; text-align: center;'>")
        html.append("<thead><tr style='background:#f2f2f2;'><th>クラス</th><th>血統</th><th>1着</th><th>2着</th><th>3着</th><th>着外</th><th>勝率</th><th>複勝率</th></tr></thead><tbody>")
        
        for _, row in pr.iterrows():
            class_name = row.get("クラス", "")
            html.append(f"<td><strong>{class_name}</strong></td>")
            html.append(f"<td>{extract_peds_name(row.get('血統', '-'))}</td>")
            html.append(f"<td>{row.get('1着', '-')}</td>")
            html.append(f"<td>{row.get('2着', '-')}</td>")
            html.append(f"<td>{row.get('3着', '-')}</td>")
            html.append(f"<td>{row.get('着外', '-')}</td>")
            html.append(f"<td>{row.get('勝率', '-')}</td>")
            html.append(f"<td>{row.get('複勝率', '-')}</td>")
            html.append("</tr>")
        
        html.append("</tbody></table>")

    # recent5
    html.append("<h4>📊 近5走成績</h4>")
    if report.get("recent5"):
        html.append("<table style='width:100%; border-collapse: collapse; text-align: center; font-size: 12px;'>")
        html.append("<thead><tr style='background:#f2f2f2;'><th>日付</th><th>開催</th><th>R</th><th>レース名</th><th>クラス</th><th>着順</th><th>人気</th><th>枠</th><th>馬番</th><th>種別</th><th>距離</th><th>馬場</th><th>タイム</th><th>着差</th><th>平均時計との差</th><th>上り</th><th>通過</th><th>馬体重</th></tr></thead><tbody>")
        
        for r in report["recent5"]:
            # 着順の色付け
            finish = r.get("result", "-")
            finish_color =  RANK_COLORS.get(finish, "#ffffff")
            finish_html = f'<td style="background-color: {finish_color}; font-weight: bold;">{finish}</td>'
            
            # 人気の色付け
            popularity = str(r.get("pops", "-"))
            pop_color = RANK_COLORS.get(popularity, "#ffffff")
            pop_html = f'<td style="background-color: {pop_color}; font-weight: bold;">{popularity}</td>'
            
            # 平均時計との差の色付け
            diff_avg = r.get("diff_avg_ms", "-")
            diff_color = get_time_diff_color(diff_avg)
            diff_html = f'<td style="color: {diff_color}; font-weight: bold;">{diff_avg}</td>'
            
            # 枠番・馬番の色付け
            waku = r.get("waku", "-")
            umaban = r.get("umaban", "-")
            waku_color = WAKU_COLORS.get(waku, "#ffffff")
            waku_html = f'<td style="background-color:{waku_color}; color:{"#fff" if waku in ["2","3","4","7"] else "#000"};">{waku}</td>'
            umaban_html =  f'<td style="background-color:{waku_color}; color:{"#fff" if waku in ["2","3","4","7"] else "#000"};">{umaban}</td>'

            # クラスの色付け
            class_name = r.get("class_name", "")
            class_bg_color = get_class_color(class_name)
            class_html = f'<td style="background-color:{class_bg_color}; padding: 2px 4px; border-radius: 3px;">{class_name}</td>'
            
            # レースタイプの色付け
            recent_race_type = r.get('race_type', '-')
            race_type_color = get_race_type_color(recent_race_type)
            race_type_html = f'<td style="background-color:{race_type_color};font-weight: bold;">{recent_race_type}</td>'
            rescent_course_len = r.get('course', '-')
            course_html = f'<td style="background-color:{race_type_color}; font-weight: bold;">{rescent_course_len}</td>'

            # 馬場状態の色付け
            ground_state = r.get('ground', '-')
            ground_state_color = get_ground_state_color(ground_state)
            ground_state_html = f'<td style="background-color:{ground_state_color};">{ground_state}</td>'


            html.append("<tr>")
            html.append(f"<td>{r.get('date', '-')}</td>")
            html.append(f"<td>{r.get('course_name', '-')}</td>")
            html.append(f"<td>{r.get('race_num', '-')}</td>")
            html.append(f"<td>{r.get('race_name', '-')}</td>")
            html.append(f"{class_html}")
            html.append(f"{finish_html}")
            html.append(f"{pop_html}")
            html.append(f"{waku_html}")
            html.append(f"{umaban_html}")
            html.append(f"{race_type_html}")
            html.append(f"{course_html}")
            html.append(f"{ground_state_html}")
            html.append(f"<td>{r.get('time_raw', '-')}</td>")
            html.append(f"<td>{r.get('diff_ms', '-')}</td>")
            html.append(f"{diff_html}")
            html.append(f"<td>{r.get('上り', '-')}</td>")
            html.append(f"<td>{r.get('通過', '-')}</td>")
            html.append(f"<td>{r.get('馬体重', '-')}</td>")
            html.append("</tr>")
        
        html.append("</tbody></table>")
    else:
        html.append("<div>直近5走データなし</div>")

    # surface summary
    html.append("<h4>🏇 芝/ダートサマリ</h4>")
    html.append("""
    <table border="1" style="border-collapse:collapse; text-align:center;">
    <thead>
        <tr>
        <th>コース</th>
        <th>最速上り</th>
        <th>平均上り</th>
        <th>平均通過位置</th>
        <th>対象レース数</th>
        </tr>
    </thead>
    <tbody>
    """)

    for surf in ["芝", "ダート"]:
        s = report.get("surface_summary", {}).get(surf, {})
        if s:
            fastest_up = s.get("fastest_up")
            fastest_info = s.get("fastest_up_info", {})
            fastest_text = "-"
            if fastest_up:
                fastest_text = f"<strong>{fastest_up}</strong> <br> {fastest_info.get('date', '-')}: {fastest_info.get('race_name', '-')} ({fastest_info.get('course_name', '-')} {fastest_info.get('course_len', '-')}m {fastest_info.get('馬場', '-')})"
            
            html.append(f"""
            <tr>
            <td>{surf}</td>
            <td>{fastest_text}</td>
            <td><strong>{safe_value(s.get('avg_up', '-'))}</strong></td>
            <td>{safe_value(s.get('avg_pass_norm', '-'))}</td>
            <td>{s.get('count', '-')}</td>
            </tr>
            """)
        else:
            html.append(f"""
            <tr>
            <td>{surf}</td>
            <td colspan="4">データなし</td>
            </tr>
            """)

    html.append("</tbody></table>")

    # same course best
    scb = report.get("same_course_best")
    html.append(f"<h4>⏱️ {place_num} {race_type}{course_len}m 持ち時計</h4>")
    if scb:
        html.append("<ul>")
        html.append(f"<li>{scb.get('date', '-')}: {scb.get('race_name', '-')} </li>")
        html.append(f"<li>タイム: <strong>{scb.get('time_str', '-')}</strong>(馬場: {scb.get('ground', '-')})</li>")
        html.append("</ul>")
    else:
        html.append("<div>同コース出走データなし</div>")

    html.append("</div>")
    return "\n".join(html)

# ---- 使い方例 ----
if __name__ == "__main__":
    # 例: raceページで「サトノシャムロック」を処理する場合
    # place_name と condition_file は呼び出し側の命名規則に合わせて指定
    place_id = 8
    horse_name = "シュバルツマサムネ"
    race_id ="202508030610"
    dates_str = "20251018"
    # race_type/course_len を与えると同コースの持ち時計も探す
    report = build_horse_report(horse_name, place_id, race_id, dates_str)
    print(report)
    # html = horse_report_to_html(report)
    # print(html)  # 先頭だけ確認