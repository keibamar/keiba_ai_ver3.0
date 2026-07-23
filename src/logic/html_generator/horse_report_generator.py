"""出走馬レポート（Forge: HTMLFactory）のHTML生成

旧 web/src/generators/horse_info.py からの移植。HTMLテンプレート・解析ロジックは
変更せず、データ取得のみ新アーキテクチャのManager層に切り替えている。
"""

import math
import re
from datetime import datetime

import numpy as np
import pandas as pd

from src.config.constants import NAME_LIST, RANK_COLORS, WAKU_COLORS
from src.managers import (
    horse_peds_dataset_manager,
    past_performance_dataset_manager,
    peds_results_dataset_manager,
    race_card_dataset_manager,
    race_info_dataset_manager,
)
from src.utils.file_utils import read_csv_or_empty

# ---- ユーティリティ ----


def time_str_to_ms(t):
    """'0:1:08.9', '1:32.7', '59.6' 等をミリ秒に変換する。解析できない場合は None"""
    if pd.isna(t) or t is None:
        return None
    s = str(t).strip()
    if s == "" or s.lower() in ["nan", "---"]:
        return None
    try:
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
            sec = float(parts[0])
            total_ms = sec * 1000
        return int(round(total_ms))
    except Exception:
        return None


def ms_to_time_str(ms):
    """ミリ秒 -> 'M:SS.s' 表示（msがNoneは '-'）"""
    if ms is None:
        return "-"
    sec = ms / 1000.0
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:04.1f}"


def normalize_passage(pass_str, heads, target_heads=18):
    """通過文字列を数値化して正規化（target_headsスケール）しリストで返す

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
        except Exception:
            out.append(None)
    return out


# ---- データ読み込み関数 ----


def load_horse_id_map():
    """data/race_info/horse_id_map.csv を horse_id/馬名 の2列に正規化して返す"""
    df = read_csv_or_empty(race_info_dataset_manager.HORSE_ID_MAP_PATH, dtype=str)
    if df.empty:
        return df
    if "horse_id" not in df.columns and "馬名" not in df.columns:
        cols = df.columns.tolist()
        if len(cols) >= 2:
            df = df.rename(columns={cols[0]: "horse_id", cols[1]: "馬名"})
        else:
            df = df.reset_index().rename(columns={"index": "horse_id", df.columns[0]: "馬名"})
    df["馬名"] = df["馬名"].astype(str).str.strip()
    return df[["horse_id", "馬名"]]


def get_avg_time(course_name, race_type, class_name, course_len, ground_state):
    """開催場名と条件から平均タイムを取得する

    Args:
        course_name (str): 開催場名（例: "中京"）
        race_type (str): レースタイプ（例: "芝" or "ダート"）
        class_name (str): クラス名（例: "3勝クラス", "未勝利", "オープン"）
        course_len (int or str): 距離（例: 1800）
        ground_state (str): 馬場状態（例: "良", "稍重", "重", "不"）

    Returns:
        float or float('nan'): 該当条件の平均タイム（存在しない場合は np.nan）
    """
    try:
        place_id = NAME_LIST.index(course_name) + 1
    except ValueError:
        print(f"❌ [要確認/エラー] 不明な開催場名: {course_name}")
        return np.nan

    df = race_info_dataset_manager.get_total_average_time_csv(place_id)
    if df.empty:
        print(f"⚠️ 平均タイムファイルが見つかりません: {course_name}")
        return np.nan

    df["course_len"] = df["course_len"].astype(str).str.strip()
    df["avg_time"] = pd.to_numeric(df["avg_time"], errors="coerce")

    if ground_state == "不良":
        ground_state = "不"
    if ground_state == "稍":
        ground_state = "稍重"
    if race_type == "ダ":
        race_type = "ダート"

    cond = (
        (df["race_type"] == str(race_type))
        & (df["course_len"] == str(course_len))
        & (df["ground_state"] == str(ground_state))
        & (df["class"] == str(class_name))
    )
    sub = df[cond]

    if sub.empty or sub["avg_time"].isna().all():
        print(f"ℹ️ [スキップ/エラーではありません] 平均タイムの該当データなし（サンプル不足）: {course_name} {race_type} {class_name} {course_len} {ground_state}")
        return np.nan

    return float(sub["avg_time"].values[0])


def get_horse_id_by_name(horse_name, map_df):
    if map_df is None or map_df.empty:
        return None
    sel = map_df[map_df["馬名"] == horse_name]
    if not sel.empty:
        return sel.iloc[0]["horse_id"]
    sel = map_df[map_df["馬名"].str.contains(horse_name, na=False)]
    if not sel.empty:
        return sel.iloc[0]["馬名"]
    return None


def load_horse_peds(horse_id):
    """data/horse/horse_peds/{horse_id}.csv を {"peds_0": ..., "peds_1": ..., ...} 形式で返す"""
    df = horse_peds_dataset_manager.get_horse_peds_csv(horse_id)
    if df.empty:
        return {}
    values = df[str(horse_id)].tolist()
    return {f"peds_{i}": str(v).strip() for i, v in enumerate(values)}


def load_peds_results(place_id, race_type, course_len, ground_state):
    """data/horse/peds_results/{place}/Total/{race_type}_{course_len}m_{ground_state}.csv を返す"""
    return peds_results_dataset_manager.get_total_peds_results_csv(place_id, race_type, course_len, ground_state)


def load_past_performance(horse_id):
    """data/horse/past_performance/{horse_id}.csv を読み、"日付_parsed"列を追加して返す"""
    df = past_performance_dataset_manager.get_past_performance_dataset(horse_id)
    if df.empty:
        return df
    if "日付" in df.columns:
        df["日付_parsed"] = pd.to_datetime(df["日付"], errors="coerce")
    df.columns = [c.strip() for c in df.columns]
    return df


# ---- 解析関数 ----


def peds_results_for_bloodline(place_id, race_type, course_len, ground_state, peds0_name):
    """指定血統（peds0_name）の peds_results が存在すれば dataframe を返す

    ファイル内の 'クラス' 列ごとにフィルタして返す（呼び出し側で利用）。
    """
    df = load_peds_results(place_id, race_type, course_len, "all")
    if df.empty:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    pattern = rf"\b{re.escape(peds0_name)}\b"

    if "血統" in df.columns:
        res = df[df["血統"].astype(str).str.contains(pattern, na=False, regex=True)]
        return res.copy()
    for col in df.columns:
        if "血統" in col:
            res = df[df[col].astype(str).str.contains(pattern, na=False, regex=True)]
            return res.copy()
    return pd.DataFrame()


def safe_value(val):
    """NaN判定して "-" に置き換える

    pd.isna()は実体がNaN/None/NaTの場合のみTrueになり、文字列"nan"（過去のCSV
    書き出し時にstr(NaN)がそのまま値として残ったもの）は素通りしてしまうため、
    文字列としての"nan"も合わせて判定する。
    """
    if val is None or val == "None":
        return "-"
    if isinstance(val, str) and val.strip().lower() == "nan":
        return "-"
    try:
        if isinstance(val, float) and math.isnan(val):
            return "-"
        if pd.isna(val):
            return "-"
    except Exception:
        pass
    return val


def recent_5_performances(horse_id, date_str):
    """past_performanceから直近5走を取得して整形して返す

    各エントリに日付、レース名、コース（距離/種別）、馬場、タイム(ms)、着差(ms)、
    上り、通過(正規化) 等を含む。
    """
    df = load_past_performance(horse_id)
    if df.empty:
        return []

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

    df = df[df["日付_parsed"] < race_day_dt]

    if df.empty:
        return []

    if df["日付_parsed"].notna().any():
        df_sorted = df.sort_values("日付_parsed", ascending=False)
    else:
        df_sorted = df.iloc[::-1].copy()

    res = []
    count = 0
    for _idx, row in df_sorted.iterrows():
        if count >= 5:
            break
        count += 1
        # race_id はDataFrameのインデックスとして格納されている
        race_id = str(_idx) if _idx is not None and str(_idx) not in ("nan", "") else row.get("race_id", "")
        date_raw = row.get("日付", "")
        waku = row.get("枠番", "")
        umaban = row.get("馬番", "")
        race_num = row.get("R", "")
        race_name = row.get("レース名", row.get("レース名", ""))
        pops = row.get("人気", "")
        try:
            pops = int(float(pops))
        except Exception:
            pops = pops
        result = row.get("着順", "")
        race_type = row.get("race_type", "")
        course_len = row.get("course_len", "")
        class_name = row.get("class", "")
        course = str(course_len)
        ground = row.get("ground_state", "")
        time_raw = row.get("タイム", "")
        t_ms = time_str_to_ms(time_raw)
        place_match = re.search(r"[0-9]*(東京|中山|阪神|京都|札幌|函館|福島|新潟|中京|小倉)[0-9]*", row.get("開催", ""))
        course_name = place_match.group(1) if place_match else ""
        diff_raw = row.get("着差", "")
        avg_time = get_avg_time(course_name, race_type, class_name, course_len, ground)
        if avg_time is not np.nan:
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

        heads = None
        if "頭 数" in row:
            try:
                heads = int(str(row.get("頭 数")).strip())
            except Exception:
                heads = None
        elif "頭数" in row:
            try:
                heads = int(str(row.get("頭数")).strip())
            except Exception:
                heads = None
        passage = row.get("通過", row.get("通過", ""))
        passage_norm = normalize_passage(passage, heads)
        if race_name is np.nan or None:
            race_date = datetime.strptime(date_raw, "%Y/%m/%d")
            df_info = race_card_dataset_manager.get_race_time_id_list_df(race_date)
            if not df_info.empty:
                match = df_info[df_info["race_id"].astype(str) == str(race_id)]
                if not match.empty:
                    race_name = str(match.iloc[0]["race_name"])

        res.append(
            {
                "race_id": safe_value(race_id),
                "date": safe_value(date_raw),
                "date_parsed": safe_value(row.get("日付_parsed", None)),
                "race_name": safe_value(race_name),
                "race_num": safe_value(race_num),
                "waku": safe_value(waku),
                "umaban": safe_value(umaban),
                "pops": safe_value(pops),
                "result": safe_value(result),
                "course_name": safe_value(course_name),
                "course": safe_value(course),
                "race_type": safe_value(race_type),
                "ground": safe_value(ground),
                "class_name": safe_value(class_name),
                "time_raw": safe_value(time_raw),
                "time_ms": safe_value(t_ms),
                "diff_avg_ms": safe_value(diff_avg_ms),
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
            }
        )
    return res


def turf_dirt_summary(horse_id, date_str):
    """past_performanceから芝/ダートごとに最速上り・平均上り・平均通過位置を計算して返す"""
    df = load_past_performance(horse_id)
    if df.empty:
        return {"芝": {}, "ダート": {}}

    df = df.copy()
    if "上り" in df.columns:
        df["上り_num"] = pd.to_numeric(df["上り"], errors="coerce")
    else:
        df["上り_num"] = pd.Series(dtype=float)
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
    df = df[df["日付_parsed"] < race_day_dt]

    surface_summary = {}
    for surface in ["芝", "ダート"]:
        search_word = surface[0]
        sub = df[df.apply(lambda r: search_word in str(r.get("race_type", "")), axis=1)]
        if sub.empty:
            surface_summary[surface] = {"fastest_up": None, "fastest_up_info": None, "avg_up": None, "avg_pass_norm": None}
            continue
        if "上り_num" in sub.columns:
            s2 = sub.copy()
            s2 = s2[s2["上り_num"].notna()]
            if not s2.empty:
                idx = s2["上り_num"].idxmin()
                fastest_row = s2.loc[idx]
                fastest_up = fastest_row["上り_num"]
                place_match = re.search(
                    r"[0-9]*(東京|中山|阪神|京都|札幌|函館|福島|新潟|中京|小倉)[0-9]*", fastest_row.get("開催", "")
                )
                course_name = place_match.group(1) if place_match else ""
                # .get(key, default)はキーが存在し値がNaNの場合はdefaultを使わずNaNを
                # そのまま返してしまう（"nan"がそのまま表示される不具合の元）ため、
                # 値自体もNaN判定して安全な値に変換する
                fastest_info = {
                    "date": safe_value(fastest_row.get("日付")),
                    "race_name": safe_value(fastest_row.get("レース名")),
                    "course_name": course_name,
                    "course_len": safe_value(fastest_row.get("course_len")),
                    "馬場": safe_value(fastest_row.get("ground_state")),
                }
            else:
                fastest_up = None
                fastest_info = None
            avg_up = s2["上り_num"].mean() if not s2.empty else None
        else:
            fastest_up = None
            fastest_info = None
            avg_up = None

        norm_list = []
        for _, r in sub.iterrows():
            heads = None
            if "頭 数" in r and not pd.isna(r.get("頭 数")):
                try:
                    heads = int(r.get("頭 数"))
                except Exception:
                    heads = None
            elif "頭数" in r and not pd.isna(r.get("頭数")):
                try:
                    heads = int(r.get("頭数"))
                except Exception:
                    heads = None
            p = r.get("通過", "")
            arr = normalize_passage(p, heads)
            norm_list.append(arr)

        avg_pass_norm = calc_average_norm_passages(norm_list)

        avg_up = safe_value(avg_up)
        try:
            avg_up = round(float(avg_up), 2)
        except Exception:
            avg_up = "-"
        avg_pass_norm = safe_value(avg_pass_norm)

        surface_summary[surface] = {
            "fastest_up": fastest_up,
            "fastest_up_info": fastest_info,
            "avg_up": avg_up,
            "avg_pass_norm": avg_pass_norm,
            "count": len(sub),
        }
    return surface_summary


def calc_average_norm_passages(norm_list):
    """正規化済み通過位置のリストを右詰めで整列し、各コーナーごとの平均を返す"""
    if not norm_list:
        return None

    max_len = max(len(x) for x in norm_list if isinstance(x, list))

    aligned = []
    for arr in norm_list:
        if not isinstance(arr, list) or len(arr) == 0:
            continue
        pad_len = max_len - len(arr)
        aligned.append([None] * pad_len + arr)

    avg_by_corner = []
    for i in range(max_len):
        vals = [row[i] for row in aligned if row[i] is not None]
        if vals:
            avg_by_corner.append(round(sum(vals) / len(vals), 1))
        else:
            avg_by_corner.append(None)

    return avg_by_corner


def same_course_best_time(horse_id, target_course_len, target_race_type, target_place_id, date_str):
    """past_performanceから同じ開催場(place_id)・距離・馬場タイプ(芝/ダート)の持ち時計を返す

    Returns:
        dict or None: {time_ms, time_str, date, race_name, ground, place_id, info_row}
    """
    df = load_past_performance(horse_id)
    if df.empty:
        return None

    df = df.fillna("").astype(str)
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
    df = df[df["日付_parsed"] < race_day_dt]

    candidates = []

    for _, row in df.iterrows():
        _raw = str(row.get("開催", ""))
        match = re.search(r"[0-9]*(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)[0-9]*", _raw)
        place_name = match.group(1) if match else ""

        if not place_name:
            continue

        try:
            place_id = int(NAME_LIST.index(place_name)) + 1
        except ValueError:
            continue

        dist_raw = str(row.get("race_type", "")).strip()
        if dist_raw.startswith("障"):
            continue
        race_type = "芝" if "芝" in dist_raw else "ダート" if "ダ" in dist_raw else ""

        dist = int(row.get("course_len", ""))
        if not race_type or not dist:
            continue

        if place_id == target_place_id and race_type == target_race_type and dist == target_course_len:
            t_ms = time_str_to_ms(row.get("タイム", ""))
            if t_ms is not None:
                candidates.append((t_ms, row, place_id))

    if not candidates:
        return None

    best_time_ms, best_row, place_id = sorted(candidates, key=lambda x: x[0])[0]

    return {
        "time_ms": best_time_ms,
        "time_str": ms_to_time_str(best_time_ms),
        "date": safe_value(best_row.get("日付")),
        "race_name": safe_value(best_row.get("レース名")),
        "ground": safe_value(best_row.get("ground_state")),
        "place_id": place_id,
        "info_row": best_row.to_dict() if hasattr(best_row, "to_dict") else {},
    }


def extract_peds_name(peds0):
    """血統名を抽出する"""
    if not peds0:
        return None

    peds0 = peds0.strip()

    match = re.match(r"^([゠-ヿー]+)", peds0)
    if match:
        return match.group(1)

    match = re.match(r"^([A-Za-z\s]+)", peds0)
    if match:
        return match.group(1).strip()

    return peds0


# ---- 統合：馬ごとの全出力を作る関数 ----


def build_horse_report(horse_name, place_id, race_id, date_str):
    """horse_name から horse_id を特定し、血統・近5走・芝ダートサマリ・持ち時計を集めて dict で返す"""
    from src.logic.html_generator import race_page_generator

    year = race_id[:4]
    race_type, course_len, ground_state, race_class = race_page_generator.get_race_info(year, place_id, race_id)
    if race_type is None and course_len is None and ground_state is None and race_class is None:
        return

    map_df = load_horse_id_map()
    hid = get_horse_id_by_name(horse_name, map_df)
    if not hid:
        return {"error": f"horse_id not found for {horse_name}"}

    peds = load_horse_peds(hid)
    peds0 = peds.get("peds_0") or peds.get("peds0") or peds.get("peds_0 ", None)
    peds0 = extract_peds_name(peds0)
    peds4_raw = peds.get("peds_4") or peds.get("peds4") or None
    peds4 = extract_peds_name(peds4_raw) if peds4_raw else None
    peds_results = None
    if peds0:
        peds_results = peds_results_for_bloodline(place_id, race_type, course_len, ground_state, peds0)
    filtered_df = peds_results[peds_results["クラス"].isin([race_class, "all"])].copy()
    if filtered_df.empty:
        print(f"⚠️ {peds0}: 該当コース({place_id}:{race_type}{course_len}) のデータがありません。")
    else:
        for idx, row in filtered_df.iterrows():
            total = int(row["1着"]) + int(row["2着"]) + int(row["3着"]) + int(row["着外"])
            win_rate = (int(row["1着"]) / total) * 100 if total > 0 else 0.0
            fukusho_rate = ((int(row["1着"]) + int(row["2着"]) + int(row["3着"])) / total) * 100 if total > 0 else 0.0
            filtered_df.at[idx, "勝率"] = round(win_rate, 1)
            filtered_df.at[idx, "複勝率"] = round(fukusho_rate, 1)

        cols = ["クラス", "血統", "1着", "2着", "3着", "着外", "勝率", "複勝率"]
        filtered_df = filtered_df[cols]

    recent5 = recent_5_performances(hid, date_str)

    surface_summary = turf_dirt_summary(hid, date_str)

    same_course_best = None
    if race_type and course_len:
        same_course_best = same_course_best_time(hid, course_len, race_type, place_id, date_str)

    return {
        "horse_name": horse_name,
        "horse_id": hid,
        "peds0": peds0,
        "peds4": peds4,
        "place_id": place_id,
        "race_type": race_type,
        "course_len": course_len,
        "ground_state": ground_state,
        "race_class": race_class,
        "peds_results": filtered_df if (isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty) else None,
        "recent5": recent5,
        "surface_summary": surface_summary,
        "same_course_best": same_course_best,
    }


# ---- 馬個別総評 ----

_CLASS_ORDER = ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "G3", "G2", "G1"]


def _class_rank(class_name):
    s = str(class_name)
    for i, c in enumerate(_CLASS_ORDER):
        if c in s:
            return i
    return -1


def _avg_pass1(surface_summary, race_type):
    """surface_summary から今回の芝/ダートの平均第1コーナー通過順(正規化値)を返す"""
    key = "芝" if race_type and "芝" in race_type else "ダート"
    s = surface_summary.get(key, {})
    norm = s.get("avg_pass_norm")
    if not norm or not isinstance(norm, list) or len(norm) == 0:
        return None
    v = norm[0]
    return float(v) if v is not None else None


_UP_THRESHOLDS = {
    # (good_threshold, poor_threshold) by class rank index matching _CLASS_ORDER
    "芝": [
        (35.5, 37.0),  # 新馬
        (35.5, 37.0),  # 未勝利
        (35.0, 36.5),  # 1勝クラス
        (34.5, 36.0),  # 2勝クラス
        (34.0, 35.5),  # 3勝クラス
        (33.5, 35.0),  # オープン
        (33.5, 35.0),  # G3
        (33.0, 34.5),  # G2
        (33.0, 34.5),  # G1
    ],
    "ダート": [
        (38.0, 39.5),  # 新馬
        (38.0, 39.5),  # 未勝利
        (37.5, 39.0),  # 1勝クラス
        (37.0, 38.5),  # 2勝クラス
        (36.5, 38.0),  # 3勝クラス
        (36.0, 37.5),  # オープン
        (36.0, 37.5),  # G3
        (35.5, 37.0),  # G2
        (35.5, 37.0),  # G1
    ],
}


def _horse_comment_html(report, ai_index, rank, popularity,
                        idx_mar=None, rank_mar=None,
                        idx_hitrate=None, rank_hitrate=None,
                        idx_value=None, rank_value=None):
    """各馬の総評文を生成する（データに特徴がある軸のみ言及）

    評価軸：
    1. AI指数の絶対水準
    2. 人気との乖離（穴候補判定）
    3. 前走からの条件変化（コース/距離/クラス）
    4. 血統のコース適性（少母数警告含む）
    5. 脚質傾向（近走vs平均の傾向変化も）
    6. 上り末脚の評価（クラス対応閾値）
    7. 持ち時計（クラス平均比較）
    8. 近5走の着順傾向
    9. MM指数が高い理由説明（rank_mar/hitrate/value いずれかが5位以内）
    """
    sentences = []

    race_type = report.get("race_type", "")
    course_len_now = report.get("course_len")
    ground_now = report.get("ground_state", "")
    race_class_now = report.get("race_class", "")
    recent5 = report.get("recent5") or []
    peds_results = report.get("peds_results")
    surface_summary = report.get("surface_summary") or {}

    # ① AI指数の絶対評価（中程度は言及しない）
    if ai_index is not None:
        if ai_index >= 70:
            sentences.append(
                f"AI指数{ai_index}点はこのモデルでトップクラスの評価で、"
                f"コース・距離別の能力面では出走馬中でも上位の水準にある。"
            )
        elif ai_index >= 63:
            sentences.append(
                f"AI指数{ai_index}点と高評価で、過去の同条件成績から能力的に一定以上の水準が認められる。"
            )
        elif ai_index <= 35:
            sentences.append(
                f"AI指数{ai_index}点と評価は低め。過去の同条件成績がこのコース・距離で苦戦傾向にある。"
            )
        elif ai_index <= 42:
            sentences.append(
                f"AI指数{ai_index}点とやや低く、今回のコース・距離への適性に課題が見られる。"
            )

    # ② 人気vs指数の乖離
    try:
        pop_int = int(popularity)
        rank_int = int(rank)
        if pop_int >= 6 and rank_int <= 2:
            sentences.append(
                f"人気（{pop_int}番人気）以上にAIが高評価しており、"
                f"オッズほどの差はないとみる穴候補の1頭。"
            )
        elif pop_int <= 2 and rank_int >= 5 and ai_index is not None and ai_index <= 50:
            sentences.append(
                f"{pop_int}番人気と支持されているが、AIスコアは低めで"
                f"人気ほどの評価はしていない。"
            )
    except (TypeError, ValueError):
        pass

    # ③ 前走からの条件変化（近5走[0]が最新走）
    if recent5:
        prev = recent5[0]
        prev_type = str(prev.get("race_type", ""))
        prev_len_raw = prev.get("course", "")
        prev_class = str(prev.get("class_name", ""))
        prev_ground = str(prev.get("ground", ""))

        # コース転換
        prev_is_turf = "芝" in prev_type
        now_is_turf = race_type and "芝" in race_type
        if prev_is_turf != now_is_turf:
            if now_is_turf:
                sentences.append(
                    f"前走はダート（{prev_len_raw}m）からの芝転向。"
                    f"過去の芝成績と血統適性がカギになる。"
                )
            else:
                sentences.append(
                    f"前走は芝（{prev_len_raw}m）からダート転向。"
                    f"ダート経験の有無と血統面での適性が注目点。"
                )
        else:
            # 距離変化
            try:
                prev_len = int(float(str(prev_len_raw)))
                now_len = int(course_len_now) if course_len_now else 0
                diff = now_len - prev_len
                if diff <= -300:
                    sentences.append(
                        f"前走{prev_len}mから{abs(diff)}m短縮。"
                        f"距離短縮で先行馬には追い風、差し馬は末脚が生きにくくなる可能性。"
                    )
                elif diff >= 300:
                    sentences.append(
                        f"前走{prev_len}mから{diff}m延長。"
                        f"スタミナ面と序盤の行き脚が問われる。"
                    )
            except (ValueError, TypeError):
                pass

        # クラス変化
        prev_cr = _class_rank(prev_class)
        now_cr = _class_rank(race_class_now)
        if prev_cr >= 0 and now_cr >= 0:
            if now_cr > prev_cr:
                sentences.append(
                    f"{prev_class}から{race_class_now}へクラス上昇。"
                    f"前走の内容と指数がクラス壁を越えられるかの判断材料になる。"
                )
            elif now_cr < prev_cr:
                sentences.append(
                    f"{prev_class}から{race_class_now}へクラス降格。"
                    f"能力的に余裕があれば上位争い可能な条件。"
                )

        # 馬場変化（良→重系 or 重系→良）
        now_heavy = ground_now in ("重", "不良", "不")
        prev_heavy = prev_ground in ("重", "不良", "不")
        if now_heavy and not prev_heavy and not sentences:
            sentences.append("前走は良馬場だったが今回は重馬場。道悪適性の有無が結果を左右する。")
        elif not now_heavy and prev_heavy and not sentences:
            sentences.append("前走は道悪だったが今回は良馬場。本来の力を発揮できる条件かどうかに注目。")

    # ④ 血統コース適性（少母数警告含む）
    if peds_results is not None and not peds_results.empty:
        all_row = peds_results[peds_results["クラス"] == "all"]
        if not all_row.empty:
            r = all_row.iloc[0]
            try:
                n1 = int(r.get("1着", 0))
                n2 = int(r.get("2着", 0))
                n3 = int(r.get("3着", 0))
                nout = int(r.get("着外", 0))
                total_n = n1 + n2 + n3 + nout
                win_r = (n1 / total_n * 100) if total_n > 0 else 0.0
                fuku_r = ((n1 + n2 + n3) / total_n * 100) if total_n > 0 else 0.0
                small_n_note = f"（参考：母数{total_n}件と少ないため信頼性に注意）" if total_n < 10 else ""
                if win_r >= 18:
                    sentences.append(
                        f"父{report.get('peds0', '')}のこのコース勝率は{win_r:.1f}%（{n1}勝/{total_n}走）と高く、"
                        f"血統面での適性は申し分ない。{small_n_note}"
                    )
                elif fuku_r >= 40:
                    sentences.append(
                        f"父{report.get('peds0', '')}の複勝率{fuku_r:.1f}%と安定感があり、"
                        f"このコースとの血統的な相性は良い。{small_n_note}"
                    )
                elif win_r > 0 and win_r <= 5 and n1 <= 2:
                    sentences.append(
                        f"父{report.get('peds0', '')}のこのコース勝率は{win_r:.1f}%と低く、"
                        f"血統面の適性には疑問が残る。{small_n_note}"
                    )
                elif total_n < 5:
                    sentences.append(
                        f"父{report.get('peds0', '')}のこのコース出走数は{total_n}件と少なく、"
                        f"血統面の適性を判断するにはデータ不足。"
                    )
            except (ValueError, TypeError):
                pass

    # ⑤ 脚質傾向（今回の芝/ダートでの平均通過順＋近走の傾向変化）
    surf_key = "芝" if race_type and "芝" in race_type else "ダート"
    p1 = _avg_pass1(surface_summary, race_type)
    if p1 is not None:
        style_label = ""
        if p1 <= 3.5:
            style_label = f"平均{p1:.1f}番手と先行タイプ。今回のコース・距離での先行有利/不利との相性が焦点。"
        elif p1 >= 9.0:
            style_label = f"平均{p1:.1f}番手と後ろから競馬するタイプ。展開と上り勝負の質次第。"

        # 近走（最大3走）の第1コーナー通過順と全体平均を比較して傾向変化を検出
        recent_pass1_list = []
        for rc in recent5[:3]:
            norm_v = rc.get("通過_norm")
            try:
                if isinstance(norm_v, list) and len(norm_v) > 0 and norm_v[0] is not None:
                    recent_pass1_list.append(float(norm_v[0]))
            except (ValueError, TypeError):
                pass
        if len(recent_pass1_list) >= 2:
            recent_p1_avg = sum(recent_pass1_list) / len(recent_pass1_list)
            diff = recent_p1_avg - p1
            if diff <= -2.5:
                trend_note = f"近走の平均通過順は{recent_p1_avg:.1f}番手と、全体平均（{p1:.1f}番手）より前で競馬するようになっている。"
                if style_label:
                    sentences.append(style_label + trend_note)
                else:
                    sentences.append(trend_note)
            elif diff >= 2.5:
                trend_note = f"近走の平均通過順は{recent_p1_avg:.1f}番手と、全体平均（{p1:.1f}番手）より後ろで競馬する傾向に変わっている。"
                if style_label:
                    sentences.append(style_label + trend_note)
                else:
                    sentences.append(trend_note)
            elif style_label:
                sentences.append(style_label)
        elif style_label:
            sentences.append(style_label)

    # ⑥ 上り末脚の評価（クラス対応の閾値）
    surf = surface_summary.get(surf_key, {})
    avg_up = surf.get("avg_up")
    surf_count = surf.get("count", 0)
    try:
        avg_up_f = float(avg_up) if avg_up not in (None, "-") else None
    except (ValueError, TypeError):
        avg_up_f = None
    if avg_up_f is not None and surf_count >= 3:
        cr = _class_rank(race_class_now)
        cr_idx = max(0, min(cr if cr >= 0 else 2, len(_UP_THRESHOLDS["芝"]) - 1))
        up_good, up_poor = _UP_THRESHOLDS.get(surf_key, _UP_THRESHOLDS["芝"])[cr_idx]
        if avg_up_f <= up_good:
            sentences.append(
                f"過去の{surf_key}レースでの平均上り{avg_up_f:.1f}秒は{race_class_now}水準では優秀で、"
                f"末脚が武器になるタイプ。上りが問われる展開なら強み。"
            )
        elif avg_up_f >= up_poor:
            sentences.append(
                f"平均上り{avg_up_f:.1f}秒は{race_class_now}水準ではやや時間がかかる傾向で、"
                f"瞬発力勝負になる展開では苦しくなりやすい。"
            )

    # ⑦ 持ち時計・コース経験（同条件馬場での平均比較）
    scb = report.get("same_course_best")
    place_id = report.get("place_id")
    course_name_for_avg = NAME_LIST[place_id - 1] if place_id and 1 <= place_id <= len(NAME_LIST) else ""
    avg_ms = np.nan
    if scb is None:
        sentences.append(
            f"このコース・距離での出走は初めてで持ち時計がなく、"
            f"コース適性は他データから推測するしかない点が評価を難しくする。"
        )
    else:
        t_str = scb.get("time_str", "-")
        scb_ground = scb.get("ground", "") or ground_now
        scb_date = scb.get("date", "")
        scb_race = scb.get("race_name", "")
        # 日付を "M/D" 形式に変換
        try:
            parts = str(scb_date).split("/")
            date_display = f"{int(parts[1])}/{int(parts[2])}"
        except Exception:
            date_display = scb_date
        race_display = f"({scb_race})" if scb_race and str(scb_race) not in ("-", "nan", "") else ""
        ground_note = f"({scb_ground})" if scb_ground else ""
        # 同じ馬場条件のクラス平均と比較（持ち時計を記録した馬場）
        if course_name_for_avg and race_type and race_class_now and course_len_now and scb_ground:
            try:
                avg_ms = get_avg_time(course_name_for_avg, race_type, race_class_now, course_len_now, scb_ground)
            except Exception:
                pass
        if not np.isnan(avg_ms):
            diff_sec = (scb["time_ms"] - avg_ms) / 1000
            avg_str = ms_to_time_str(int(avg_ms))
            if diff_sec <= -0.5:
                sentences.append(
                    f"{date_display}{race_display}で{t_str}{ground_note}の持ち時計。"
                    f"{scb_ground}のクラス平均{avg_str}より{abs(diff_sec):.1f}秒速く、十分な水準。"
                )
            elif diff_sec <= 0.5:
                sentences.append(
                    f"{date_display}{race_display}で{t_str}{ground_note}の持ち時計。"
                    f"{scb_ground}のクラス平均{avg_str}とほぼ同等。"
                )
            else:
                sentences.append(
                    f"{date_display}{race_display}で{t_str}{ground_note}の持ち時計があるが、"
                    f"{scb_ground}のクラス平均{avg_str}より{diff_sec:.1f}秒遅く、タイムを縮める必要あり。"
                )
        else:
            sentences.append(
                f"{date_display}{race_display}で{t_str}{ground_note}の持ち時計があり、コース経験は十分。"
            )

    # ⑧ 近5走の着順傾向（3走以上のデータで傾きが明確なときだけ）
    results_numeric = []
    for r in recent5:
        val = r.get("result", "-")
        try:
            results_numeric.append(int(str(val).strip()))
        except (ValueError, TypeError):
            pass
    if len(results_numeric) >= 3:
        # 新しい順（recent5[0]が最新）なので古い順に並べ替えて傾き計算
        vals = list(reversed(results_numeric))
        n = len(vals)
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(vals) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, vals))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = num / den if den > 0 else 0.0
        if slope < -1.0:
            sentences.append(
                f"近{n}走の着順が上向き傾向（平均{y_mean:.1f}着）で、"
                f"充実期にある可能性が高い。"
            )
        elif slope > 1.0:
            sentences.append(
                f"近{n}走の着順が下降気味（平均{y_mean:.1f}着）で、"
                f"フォームの立て直しが課題。今回の巻き返しに期待できるかが焦点。"
            )

    # ⑨ MM指数高評価の理由説明（いずれかのモデルでrank ≤ 5）
    high_ranks = []
    if rank_mar is not None and rank_mar <= 5:
        high_ranks.append(f"MAR({rank_mar}位)")
    if rank_hitrate is not None and rank_hitrate <= 5:
        high_ranks.append(f"的中率({rank_hitrate}位)")
    if rank_value is not None and rank_value <= 5:
        high_ranks.append(f"回収率({rank_value}位)")
    if high_ranks:
        reasons = []
        # 血統適性（具体的な勝率と出走数）
        if peds_results is not None and not peds_results.empty:
            all_row = peds_results[peds_results["クラス"] == "all"]
            if not all_row.empty:
                rr = all_row.iloc[0]
                try:
                    rn1 = int(rr.get("1着", 0))
                    rn_total = rn1 + int(rr.get("2着", 0)) + int(rr.get("3着", 0)) + int(rr.get("着外", 0))
                    r_win = (rn1 / rn_total * 100) if rn_total > 0 else 0
                    if r_win >= 15:
                        reasons.append(f"父{report.get('peds0', '')}のコース勝率{r_win:.0f}%（{rn1}勝/{rn_total}走）")
                except (ValueError, TypeError):
                    pass
        # 持ち時計（具体的な日付・馬場・クラス平均との比較）
        if scb is not None:
            try:
                parts = str(scb.get("date", "")).split("/")
                scb_date_disp = f"{int(parts[1])}/{int(parts[2])}"
            except Exception:
                scb_date_disp = scb.get("date", "")
            scb_g = scb.get("ground", "") or ground_now
            scb_rname = scb.get("race_name", "")
            scb_rname_disp = f"({scb_rname})" if scb_rname and str(scb_rname) not in ("-", "nan", "") else ""
            if not np.isnan(avg_ms) and (scb["time_ms"] - avg_ms) / 1000 <= -0.3:
                diff_disp = abs((scb["time_ms"] - avg_ms) / 1000)
                reasons.append(
                    f"{scb_date_disp}{scb_rname_disp}で{scb['time_str']}({scb_g})の持ち時計"
                    f"（クラス平均より{diff_disp:.1f}秒速い）"
                )
            elif not np.isnan(avg_ms) and (scb["time_ms"] - avg_ms) / 1000 <= 0.3:
                reasons.append(
                    f"{scb_date_disp}{scb_rname_disp}で{scb['time_str']}({scb_g})の持ち時計"
                    f"（クラス平均並み）"
                )
        # 上り末脚（クラス基準値を明記、何走で使えているかを表示）
        if avg_up_f is not None and surf_count >= 2:
            cr2 = _class_rank(race_class_now)
            cr2_idx = max(0, min(cr2 if cr2 >= 0 else 2, len(_UP_THRESHOLDS["芝"]) - 1))
            thr_good, _ = _UP_THRESHOLDS.get(surf_key, _UP_THRESHOLDS["芝"])[cr2_idx]
            # 近走で上りが良かったレース数
            good_up_count = 0
            for rc in recent5:
                try:
                    up_v = float(str(rc.get("上り", "")).replace("-", ""))
                    if up_v <= thr_good:
                        good_up_count += 1
                except (ValueError, TypeError):
                    pass
            if avg_up_f <= thr_good:
                reasons.append(
                    f"{surf_key}平均上り{avg_up_f:.1f}秒"
                    f"（{race_class_now}目安{thr_good:.1f}秒以下、近{len(recent5[:5])}走で{good_up_count}走が基準クリア）"
                )
        # 近走好成績（具体的な日付・着順）
        good_recent_strs = []
        for rc in recent5[:4]:
            res = str(rc.get("result", "")).strip()
            if res in ("1", "2", "3"):
                d = rc.get("date", "")
                rn = rc.get("race_name", "")
                try:
                    dp = d.split("/")
                    d_disp = f"{int(dp[1])}/{int(dp[2])}"
                except Exception:
                    d_disp = d
                rn_disp = f"({rn})" if rn and str(rn) not in ("-", "nan", "") else ""
                good_recent_strs.append(f"{d_disp}{rn_disp}{res}着")
        if len(good_recent_strs) >= 2:
            reasons.append("近走好走：" + "、".join(good_recent_strs[:3]))
        elif good_recent_strs:
            reasons.append(f"近走{good_recent_strs[0]}")
        # 理由がない場合でも手持ちのデータで具体的に説明
        if not reasons:
            fallback_parts = []
            if avg_up_f is not None and surf_count >= 2:
                cr2 = _class_rank(race_class_now)
                cr2_idx = max(0, min(cr2 if cr2 >= 0 else 2, len(_UP_THRESHOLDS["芝"]) - 1))
                thr_g, _ = _UP_THRESHOLDS.get(surf_key, _UP_THRESHOLDS["芝"])[cr2_idx]
                fallback_parts.append(f"{surf_key}平均上り{avg_up_f:.1f}秒（目安{thr_g:.1f}秒、{surf_count}走実績）")
            if scb is not None:
                scb_g = scb.get("ground", "")
                fallback_parts.append(f"持ち時計{scb['time_str']}（{scb_g}）")
            if not fallback_parts:
                fallback_parts.append("過去成績とタイムの総合評価")
            reasons = fallback_parts
        sentences.append(
            f"{'・'.join(high_ranks)}でモデル高評価。"
            f"主な要因：{'、'.join(reasons)}。"
        )

    if not sentences:
        return ""

    items = "".join(f"<li>{s}</li>" for s in sentences)
    return f'<ul class="horse-comment">{items}</ul>\n'


# ---- HTML 整形 ----


def get_time_diff_color(diff_str):
    """平均勝ち時計との差で色を返す（マイナス=赤、0〜0.2秒=オレンジ、それ以上=黒）"""
    try:
        diff_str = str(diff_str).strip()
        if not diff_str or diff_str == "-":
            return "black"

        diff_str_clean = diff_str.replace("秒", "").strip()
        diff_val = float(diff_str_clean)

        if diff_val < 0:
            return "red"
        elif 0 <= diff_val <= 0.2:
            return "orange"
        else:
            return "black"
    except Exception:
        return "black"


def get_class_color(class_name):
    """クラスに基づいて背景色を返す"""
    class_colors = {
        "未勝利": "#fff0f0",
        "新馬": "#ffe6e6",
        "1勝クラス": "#ffcccc",
        "2勝クラス": "#ffb3b3",
        "3勝クラス": "#ff9999",
        "オープン": "#ff8080",
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


def _lookup_recent_indices(race_id, umaban, race_date=None):
    """近走レース(race_id)でのこの馬(umaban)のMAR/的中率/回収率指数を返す

    race_date: "YYYY/MM/DD" 形式の実際のレース日（ディレクトリ探索に使用）
               省略時は race_id[:8] から推測するが race_id がJRA形式の場合は不正確

    Returns: (idx_mar, rank_mar, idx_hitrate, rank_hitrate, idx_value, rank_value) or all None
    """
    if not race_id or str(race_id) in ("-", "nan", ""):
        return None, None, None, None, None, None
    try:
        rid = str(race_id).strip()
        if len(rid) < 12:
            return None, None, None, None, None, None
        from datetime import datetime
        # race_date が渡されていればそれを優先（race_id[:8]はJRA開催番号で実際の日付ではない）
        if race_date and str(race_date) not in ("-", "nan", ""):
            try:
                race_day = datetime.strptime(str(race_date).strip(), "%Y/%m/%d").date()
            except ValueError:
                try:
                    race_day = datetime.strptime(str(race_date).strip(), "%Y%m%d").date()
                except ValueError:
                    race_day = datetime.strptime(rid[:8], "%Y%m%d").date()
        else:
            race_day = datetime.strptime(rid[:8], "%Y%m%d").date()
        card = race_card_dataset_manager.get_race_cards(race_day, rid)
        if card.empty or "馬番" not in card.columns:
            return None, None, None, None, None, None
        mask = card["馬番"].astype(str) == str(umaban)
        if not mask.any():
            return None, None, None, None, None, None
        row = card[mask].iloc[0]
        def _f(col):
            v = row.get(col, None)
            try:
                return float(v) if v is not None and str(v) not in ("nan", "") else None
            except Exception:
                return None
        def _i(col):
            v = row.get(col, None)
            try:
                return int(float(v)) if v is not None and str(v) not in ("nan", "") else None
            except Exception:
                return None
        idx_mar = _f("idx_mar")
        rank_mar = _i("rank_mar")
        idx_hr = _f("idx_hitrate")
        rank_hr = _i("rank_hitrate")
        idx_val = _f("idx_value")
        rank_val = _i("rank_value")
        # multi-model列がなければ旧 score/rank を MAR 列にフォールバック
        if idx_mar is None and idx_hr is None and idx_val is None:
            idx_score = _f("score")
            rank_score = _i("rank")
            if idx_score is not None:
                return idx_score, rank_score, None, None, None, None
        return idx_mar, rank_mar, idx_hr, rank_hr, idx_val, rank_val
    except Exception:
        return None, None, None, None, None, None


def _mini_rank(rnk):
    """近走指数用の小さなランク表示"""
    if rnk is None:
        return ""
    color = "#d32f2f" if rnk <= 5 else "#888"
    return f'<sub style="font-size:0.65em;color:{color};font-weight:bold;">({rnk})</sub>'


def horse_report_to_html(report, ai_index=None, rank=None, popularity=None,
                          idx_mar=None, rank_mar=None,
                          idx_hitrate=None, rank_hitrate=None,
                          idx_value=None, rank_value=None):
    """build_horse_report の出力からスタイル付きHTMLを作る

    色付けロジック:
    - 着順、人気: 1着=黄色、2着=水色、3着=オレンジ
    - 平均時計との差: マイナス=赤、0.2秒以内=オレンジ、それ以上=黒
    - クラス: 未勝利→新馬→1勝→2勝→3勝→オープンで色が濃くなる
    - 枠番・馬番: 数字のみに色付け
    """
    if "error" in report:
        return f"<div class='horse-report error'>{report['error']}</div>"

    html = []
    html.append("<div class='horse-report' style='padding: 10px; background: #fafafa; border: 1px solid #ddd;'>")

    comment = _horse_comment_html(report, ai_index, rank, popularity,
                                  idx_mar=idx_mar, rank_mar=rank_mar,
                                  idx_hitrate=idx_hitrate, rank_hitrate=rank_hitrate,
                                  idx_value=idx_value, rank_value=rank_value)
    if comment:
        html.append('<div class="horse-comment-wrap">')
        html.append(comment)
        html.append("</div>")

    p0 = report.get("peds0") or ""
    p4 = report.get("peds4") or ""
    if p0 and p4:
        html.append(f"<h4>血統: 父 <strong>{p0}</strong> / 母父 <strong>{p4}</strong></h4>")
    elif p0:
        html.append(f"<h4>血統: 父 <strong>{p0}</strong></h4>")
    else:
        html.append("<h4>血統: -</h4>")

    race_type = report.get("race_type", "-")
    course_len = report.get("course_len", "-")
    ground_state = report.get("ground_state", "-")
    place_id = report.get("place_id", "-")
    place_num = NAME_LIST[place_id - 1]

    pr = report.get("peds_results")
    if pr is None or (isinstance(pr, pd.DataFrame) and pr.empty):
        html.append("<div>血統データなし</div>")
    else:
        html.append(f"<h4>🧬 {place_num} {race_type}{course_len}m ({ground_state})</h4>")
        html.append('<p class="chakudo-legend">着度数 (1着,2着,3着,着外) ／ バーにマウスを合わせると内訳の割合（2着・3着は累積割合も）を確認できます</p>')
        html.append('<div class="chakudo-chart">')

        for _, row in pr.iterrows():
            class_name = row.get("クラス", "")
            peds_name  = extract_peds_name(row.get("血統", "-"))
            n1   = int(row.get("1着",  0) or 0)
            n2   = int(row.get("2着",  0) or 0)
            n3   = int(row.get("3着",  0) or 0)
            nout = int(row.get("着外", 0) or 0)
            total = n1 + n2 + n3 + nout
            if total == 0:
                continue
            w1   = n1   / total * 100
            w2   = n2   / total * 100
            w3   = n3   / total * 100
            wout = nout / total * 100
            win_rate   = round(n1 / total * 100, 1)
            cumul2     = round((n1 + n2) / total * 100, 1)
            place_rate = round((n1 + n2 + n3) / total * 100, 1)
            out_rate   = round(nout / total * 100, 1)
            seg1   = (f'<span class="chakudo-segment seg-1st" style="width:{w1:.2f}%">{n1}'
                      f'<span class="chakudo-tooltip">1着 {n1}回 ／ 勝率 {win_rate}%</span></span>')
            seg2   = (f'<span class="chakudo-segment seg-2nd" style="width:{w2:.2f}%">{n2}'
                      f'<span class="chakudo-tooltip">2着 {n2}回 ／ 累積 {cumul2}%</span></span>')
            seg3   = (f'<span class="chakudo-segment seg-3rd" style="width:{w3:.2f}%">{n3}'
                      f'<span class="chakudo-tooltip">3着 {n3}回 ／ 複勝率 {place_rate}%</span></span>')
            segout = (f'<span class="chakudo-segment seg-out" style="width:{wout:.2f}%">{nout}'
                      f'<span class="chakudo-tooltip">着外 {nout}回 ／ 着外率 {out_rate}%</span></span>')
            html.append(
                f'<div class="chakudo-row">'
                f'<span class="chakudo-label" style="width:56px;text-align:left;font-size:0.85em;">{class_name}</span>'
                f'<span class="chakudo-label peds-label">{peds_name}</span>'
                f'<span class="chakudo-bar-track">{seg1}{seg2}{seg3}{segout}</span>'
                f'<span class="chakudo-value">n={total}</span>'
                f'</div>'
            )
        html.append("</div>")

    html.append("<h4>📊 近5走成績</h4>")
    if report.get("recent5"):
        html.append('<div class="table-wrap">')
        html.append("<table style='border-collapse: collapse; text-align: center; font-size: 12px;'>")
        html.append(
            "<thead><tr style='background:#f2f2f2;'><th>日付</th><th>開催</th><th>R</th><th>レース名</th>"
            "<th>クラス</th><th>着順</th><th>人気</th><th>枠</th><th>馬番</th><th>種別</th><th>距離</th>"
            "<th>馬場</th><th>タイム</th><th>着差</th><th>平均差</th><th>上り</th><th>通過</th><th>馬体重</th>"
            "<th style='background:#6a1b9a;color:white;' title='MAR推奨指数（2025以前は旧スコア）'>MAR</th>"
            "<th style='background:#9c27b0;color:white;'>的中率</th>"
            "<th style='background:#ce93d8;color:#4a148c;'>回収率</th>"
            "</tr></thead><tbody>"
        )

        for r in report["recent5"]:
            # 近走指数を race_card CSV から取得（race_dateはディレクトリ名に使うため必須）
            _rid  = r.get("race_id", "")
            _uban = r.get("umaban", "")
            _rdate = r.get("date", "")
            _im, _rm, _ih, _rh, _iv, _rv = _lookup_recent_indices(_rid, _uban, race_date=_rdate)
            finish = r.get("result", "-")
            finish_color = RANK_COLORS.get(finish, "#ffffff")
            finish_html = f'<td style="background-color: {finish_color}; font-weight: bold;">{finish}</td>'

            popularity = str(r.get("pops", "-"))
            pop_color = RANK_COLORS.get(popularity, "#ffffff")
            pop_html = f'<td style="background-color: {pop_color}; font-weight: bold;">{popularity}</td>'

            diff_avg = r.get("diff_avg_ms", "-")
            diff_color = get_time_diff_color(diff_avg)
            diff_html = f'<td style="color: {diff_color}; font-weight: bold;">{diff_avg}</td>'

            waku = r.get("waku", "-")
            umaban = r.get("umaban", "-")
            waku_color = WAKU_COLORS.get(waku, "#ffffff")
            waku_html = f'<td style="background-color:{waku_color}; color:{"#fff" if waku in ["2", "3", "4", "7"] else "#000"};">{waku}</td>'
            umaban_html = f'<td style="background-color:{waku_color}; color:{"#fff" if waku in ["2", "3", "4", "7"] else "#000"};">{umaban}</td>'

            class_name = r.get("class_name", "")
            class_bg_color = get_class_color(class_name)
            class_html = f'<td style="background-color:{class_bg_color}; padding: 2px 4px; border-radius: 3px;">{class_name}</td>'

            recent_race_type = r.get("race_type", "-")
            race_type_color = get_race_type_color(recent_race_type)
            race_type_html = f'<td style="background-color:{race_type_color};font-weight: bold;">{recent_race_type}</td>'
            rescent_course_len = r.get("course", "-")
            course_html = f'<td style="background-color:{race_type_color}; font-weight: bold;">{rescent_course_len}</td>'

            ground_state = r.get("ground", "-")
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
            # 近走AI指数セル
            def _idx_cell(val, rnk, bg_main="#FFFDE7"):
                if val is None:
                    return "<td style='color:#ccc;'>-</td>"
                bg = bg_main if rnk and rnk <= 3 else ""
                bg_css = f"background-color:{bg};" if bg else ""
                rc = "#d32f2f" if rnk and rnk <= 5 else "#888"
                rs = f'<sub style="font-size:0.65em;color:{rc};font-weight:bold;">({rnk})</sub>' if rnk else ""
                return f'<td style="{bg_css}font-weight:{"bold" if rnk and rnk<=3 else "normal"};">{val:.1f}{rs}</td>'
            html.append(_idx_cell(_im, _rm, "#FFF176"))
            html.append(_idx_cell(_ih, _rh, "#FFF176"))
            html.append(_idx_cell(_iv, _rv, "#E0F7FA"))
            html.append("</tr>")

        html.append("</tbody></table>")
        html.append("</div>")
    else:
        html.append("<div>直近5走データなし</div>")

    html.append("<h4>🏇 芝/ダートサマリ</h4>")
    html.append('<div class="table-wrap">')
    html.append(
        """
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
    """
    )

    for surf in ["芝", "ダート"]:
        s = report.get("surface_summary", {}).get(surf, {})
        if s:
            fastest_up = s.get("fastest_up")
            fastest_info = s.get("fastest_up_info", {})
            fastest_text = "-"
            if fastest_up:
                fastest_text = (
                    f"<strong>{fastest_up}</strong> <br> {fastest_info.get('date', '-')}: "
                    f"{fastest_info.get('race_name', '-')} ({fastest_info.get('course_name', '-')} "
                    f"{fastest_info.get('course_len', '-')}m {fastest_info.get('馬場', '-')})"
                )

            html.append(
                f"""
            <tr>
            <td>{surf}</td>
            <td>{fastest_text}</td>
            <td><strong>{safe_value(s.get('avg_up', '-'))}</strong></td>
            <td>{safe_value(s.get('avg_pass_norm', '-'))}</td>
            <td>{s.get('count', '-')}</td>
            </tr>
            """
            )
        else:
            html.append(
                f"""
            <tr>
            <td>{surf}</td>
            <td colspan="4">データなし</td>
            </tr>
            """
            )

    html.append("</tbody></table>")
    html.append("</div>")

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
