"""新馬戦専用 LightGBM 学習データセット生成

走歴なしの新馬戦に特化した特徴量セット。

Features (SHINBA_FEATURES 列定義・29列):
  父統計 (5列):
    sire_course_win    父の新馬戦(同race_type×距離帯)勝率
    sire_course_place  父の新馬戦(同race_type×距離帯)複勝率
    sire_course_runs   父の新馬戦(同race_type×距離帯)出走数
    sire_global_win    父の新馬戦(全コース)勝率
    sire_global_place  父の新馬戦(全コース)複勝率
  母父(BMS)統計 (5列):
    bms_course_win     母父の新馬戦(同race_type×距離帯)勝率
    bms_course_place   母父の新馬戦(同race_type×距離帯)複勝率
    bms_course_runs    母父の新馬戦(同race_type×距離帯)出走数
    bms_global_win     母父の新馬戦(全コース)勝率
    bms_global_place   母父の新馬戦(全コース)複勝率
  同牝系統計 (3列):
    dam_shinba_win     同牝系(母)の新馬戦勝率 (runs<3時はNaN)
    dam_shinba_place   同牝系(母)の新馬戦複勝率 (runs<3時はNaN)
    dam_shinba_runs    同牝系(母)の新馬戦出走数 (runs<3時はNaN)
  父父(祖父)統計 (2列):
    grandsire_win      父父の新馬戦(全コース)勝率
    grandsire_place    父父の新馬戦(全コース)複勝率
  近親交配クロス (1列):
    cross_score        近親交配スコア（父系・母系共通祖先を4代まで検索, Σ1/2^(gS+gD)）
  騎手統計 (2列):
    jockey_shinba_win    騎手の新馬戦勝率
    jockey_shinba_place  騎手の新馬戦複勝率
  調教師統計 (2列):
    trainer_shinba_win   調教師の新馬戦勝率
    trainer_shinba_place 調教師の新馬戦複勝率
  レース・馬属性 (9列):
    place_id      開催場ID (1-10)
    race_type_code 0=芝 / 1=ダート
    course_len    コース距離 (m)
    waku          枠番
    umaban        馬番
    kinryo        斤量
    sex_code      0=牡/セン / 1=牝
    horse_weight  馬体重 (kg)
    birth_month   誕生月 (1〜12、不明時はNaN)
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import paths
from src.config.constants import PLACE_LIST
from src.managers import horse_peds_dataset_manager, horse_profile_dataset_manager, race_result_dataset_manager

SHINBA_FEATURES = [
    # 父統計 (5列)
    "sire_course_win",
    "sire_course_place",
    "sire_course_runs",
    "sire_global_win",
    "sire_global_place",
    # 母父(BMS)統計 (5列)
    "bms_course_win",
    "bms_course_place",
    "bms_course_runs",
    "bms_global_win",
    "bms_global_place",
    # 同牝系(母)統計 (3列)
    "dam_shinba_win",
    "dam_shinba_place",
    "dam_shinba_runs",
    # 父父(祖父)統計 (2列)
    "grandsire_win",
    "grandsire_place",
    # 近親交配クロス (1列)
    "cross_score",
    # 騎手統計 (2列)
    "jockey_shinba_win",
    "jockey_shinba_place",
    # 調教師統計 (2列)
    "trainer_shinba_win",
    "trainer_shinba_place",
    # レース・馬属性 (9列)
    "place_id",
    "race_type_code",
    "course_len",
    "waku",
    "umaban",
    "kinryo",
    "sex_code",
    "horse_weight",
    "birth_month",
]

# 距離帯（父×コース統計を帯単位でまとめてサンプル数を稼ぐ）
_DIST_BANDS = [(0, 1400), (1400, 1900), (1900, 2200), (2200, 99999)]

# 同牝系統計の最小出走数（これ未満はNaN）
_DAM_MIN_RUNS = 3


def _dist_band(length):
    length = int(length)
    for lo, hi in _DIST_BANDS:
        if lo <= length < hi:
            return f"{lo}_{hi}"
    return "2200_99999"


def _parse_date_jp(date_str):
    """'2019年01月05日' → pd.Timestamp"""
    try:
        s = re.sub(r"[年月]", "-", str(date_str)).replace("日", "").strip()
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def _parse_weight(weight_str):
    """'480(+4)' → 480.0"""
    try:
        m = re.match(r"(\d+)", str(weight_str))
        return float(m.group(1)) if m else np.nan
    except Exception:
        return np.nan


def _parse_kinryo(v):
    try:
        return float(str(v).strip())
    except Exception:
        return np.nan


def _sex_code(seirei_str):
    """性齢文字列 → 0=牡/セン, 1=牝"""
    s = str(seirei_str)
    return 1.0 if s.startswith("牝") else 0.0


def _load_all_shinba_results():
    """全場の新馬戦レース結果を読み込んで結合する"""
    files = glob.glob(
        os.path.join(paths.RACE_RESULT_DATA_PATH, "*", "*_race_results.csv")
    )
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str, index_col=0)
            df.index.name = "race_id"
            df = df.reset_index()
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    shinba = all_df[all_df["class"] == "新馬"].copy()
    return shinba


def _safe_ped(val):
    """peds値をstr変換（NaN/None/空は空文字に）"""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    s = str(val).strip()
    return s if s and s.lower() != "nan" else ""


def _compute_cross_score(peds_list, max_gen=4):
    """近親交配クロススコアを計算する。
    父系と母系の共通祖先を max_gen 代まで走査し、
    score = Σ 1/2^(gen_sire + gen_dam) を返す（近いクロスほど高スコア）。
    """
    if not peds_list:
        return 0.0

    sire_gen = {}  # ancestor_name → sire 側での最小代数
    dam_gen  = {}  # ancestor_name → dam 側での最小代数

    def _traverse(node_idx, gen, result_dict):
        if gen > max_gen or node_idx >= len(peds_list):
            return
        name = _safe_ped(peds_list[node_idx])
        if name and gen < result_dict.get(name, 99):
            result_dict[name] = gen
        _traverse(2 * node_idx + 2, gen + 1, result_dict)
        _traverse(2 * node_idx + 3, gen + 1, result_dict)

    _traverse(0, 1, sire_gen)
    _traverse(1, 1, dam_gen)

    score = 0.0
    for name, sg in sire_gen.items():
        if name in dam_gen:
            score += 1.0 / (2 ** (sg + dam_gen[name]))
    return score


def _fetch_peds_map(horse_ids):
    """horse_id → (sire, dam, grandsire, bms, birth_month, cross_score) のマップを返す
    peds_0=父, peds_1=母, peds_2=父父, peds_4=母父(BMS)
    birth_month=誕生月(1〜12、不明時はNaN), cross_score=近親交配スコア
    """
    peds_map = {}
    for hid in tqdm(horse_ids, desc="horse_peds読込"):
        try:
            peds_data = horse_peds_dataset_manager.get_horse_peds_dataset(str(hid))
            peds_list = peds_data[str(hid)].tolist()
            sire        = _safe_ped(peds_list[0]) if len(peds_list) > 0 else ""
            dam         = _safe_ped(peds_list[1]) if len(peds_list) > 1 else ""
            grandsire   = _safe_ped(peds_list[2]) if len(peds_list) > 2 else ""
            bms         = _safe_ped(peds_list[4]) if len(peds_list) > 4 else ""
            cross_score = _compute_cross_score(peds_list)
        except Exception:
            sire, dam, grandsire, bms, cross_score = "", "", "", "", 0.0
        birth_month = horse_profile_dataset_manager.get_birth_month(str(hid))
        peds_map[str(hid)] = (sire, dam, grandsire, bms, birth_month, cross_score)
    return peds_map


def _build_cumulative_stats(df, group_cols, win_col="_is_win", place_col="_is_place"):
    """
    group_cols でグループ化した累積勝率・複勝率を計算（時系列リーク防止）。
    df は date_dt でソート済みであること。
    """
    win_arr   = np.full(len(df), np.nan)
    place_arr = np.full(len(df), np.nan)
    runs_arr  = np.zeros(len(df), dtype=float)

    for _, grp in df.groupby(group_cols, sort=False):
        idx   = grp.index.tolist()
        wins  = grp[win_col].values.astype(float)
        plcs  = grp[place_col].values.astype(float)
        counts = np.arange(1, len(idx) + 1, dtype=float)
        cum_w = np.cumsum(wins)
        cum_p = np.cumsum(plcs)
        if len(idx) > 1:
            win_arr[idx[1:]]   = cum_w[:-1] / counts[:-1]
            place_arr[idx[1:]] = cum_p[:-1] / counts[:-1]
            runs_arr[idx[1:]]  = counts[:-1]

    return win_arr, place_arr, runs_arr


def make_training_dataset():
    """
    全新馬戦データから学習用 DataFrame を生成して返す。

    Returns:
        tuple: (df_feat, groups, labels)
          df_feat  : SHINBA_FEATURES 列の DataFrame（行=馬）
          groups   : レースごとの出走頭数リスト (LambdaRank用)
          labels   : 各馬の label (1着=3, 2着=2, 3着=1, 4着以下=0)
    """
    print("新馬戦レース結果を読み込み中...")
    shinba = _load_all_shinba_results()
    if shinba.empty:
        raise RuntimeError("新馬戦データが見つかりません")
    print(f"  → {len(shinba)}件")

    # 血統情報 + 誕生月 + クロスを取得
    unique_ids = shinba["horse_id"].dropna().unique().tolist()
    print(f"  → 馬 {len(unique_ids)}頭の血統・誕生月・クロスを取得中...")
    peds_map = _fetch_peds_map(unique_ids)
    _empty = ("", "", "", "", None, 0.0)
    shinba["sire"]        = shinba["horse_id"].map(lambda x: peds_map.get(str(x), _empty)[0])
    shinba["dam"]         = shinba["horse_id"].map(lambda x: peds_map.get(str(x), _empty)[1])
    shinba["grandsire"]   = shinba["horse_id"].map(lambda x: peds_map.get(str(x), _empty)[2])
    shinba["bms"]         = shinba["horse_id"].map(lambda x: peds_map.get(str(x), _empty)[3])
    shinba["birth_month"] = shinba["horse_id"].map(
        lambda x: float(peds_map.get(str(x), _empty)[4]) if peds_map.get(str(x), _empty)[4] is not None else np.nan
    )
    shinba["cross_score"] = shinba["horse_id"].map(
        lambda x: float(peds_map.get(str(x), _empty)[5])
    )

    # 日付パース・場所・コード変換
    shinba["date_dt"]      = shinba["date"].apply(_parse_date_jp)
    shinba["place_id_int"] = shinba["race_id"].apply(
        lambda r: int(str(r)[4:6]) if str(r)[4:6].isdigit() else -1
    )
    shinba["race_type_code"] = (shinba["race_type"] == "ダート").astype(float)
    shinba["dist_band"]    = shinba["course_len"].apply(
        lambda x: _dist_band(x) if str(x).isdigit() else "unknown"
    )
    shinba["horse_weight"] = shinba["馬体重"].apply(_parse_weight)
    shinba["kinryo_f"]     = shinba["斤量"].apply(_parse_kinryo)
    shinba["sex_code"]     = shinba["性齢"].apply(_sex_code)
    shinba["waku_int"]     = pd.to_numeric(shinba["枠番"], errors="coerce")
    shinba["umaban_int"]   = pd.to_numeric(shinba["馬番"], errors="coerce")
    shinba["course_len_f"] = pd.to_numeric(shinba["course_len"], errors="coerce")

    # 着順 → label / is_win / is_place
    def _rank_to_label(v):
        s = re.sub(r"\D", "", str(v))
        if not s:
            return np.nan
        r = int(s)
        if r == 1:
            return 3.0
        if r == 2:
            return 2.0
        if r == 3:
            return 1.0
        return 0.0

    def _rank_int(v):
        s = re.sub(r"\D", "", str(v))
        return int(s) if s else None

    shinba["label"]    = shinba["着順"].apply(_rank_to_label)
    shinba["_rank_i"]  = shinba["着順"].apply(_rank_int)
    shinba["_is_win"]  = (shinba["_rank_i"] == 1).astype(float)
    shinba["_is_plc"]  = shinba["_rank_i"].apply(
        lambda r: 1.0 if r is not None and r <= 3 else 0.0
    )

    # NaN label 行を除外
    shinba = shinba.dropna(subset=["label", "date_dt"]).copy()
    shinba = shinba.sort_values("date_dt").reset_index(drop=True)

    print("累積父統計（全コース）を計算中...")
    g_win, g_plc, _ = _build_cumulative_stats(shinba, ["sire"], "_is_win", "_is_plc")
    shinba["sire_global_win"]   = g_win
    shinba["sire_global_place"] = g_plc

    print("累積父統計（race_type×距離帯）を計算中...")
    c_win, c_plc, c_runs = _build_cumulative_stats(
        shinba, ["sire", "race_type", "dist_band"], "_is_win", "_is_plc"
    )
    shinba["sire_course_win"]   = c_win
    shinba["sire_course_place"] = c_plc
    shinba["sire_course_runs"]  = c_runs

    print("累積母父(BMS)統計（全コース）を計算中...")
    bg_win, bg_plc, _ = _build_cumulative_stats(shinba, ["bms"], "_is_win", "_is_plc")
    shinba["bms_global_win"]   = bg_win
    shinba["bms_global_place"] = bg_plc

    print("累積母父(BMS)統計（race_type×距離帯）を計算中...")
    bc_win, bc_plc, bc_runs = _build_cumulative_stats(
        shinba, ["bms", "race_type", "dist_band"], "_is_win", "_is_plc"
    )
    shinba["bms_course_win"]   = bc_win
    shinba["bms_course_place"] = bc_plc
    shinba["bms_course_runs"]  = bc_runs

    print("累積同牝系(母)統計を計算中...")
    d_win, d_plc, d_runs = _build_cumulative_stats(shinba, ["dam"], "_is_win", "_is_plc")
    # runs < _DAM_MIN_RUNS は NaN
    mask_dam = d_runs < _DAM_MIN_RUNS
    d_win[mask_dam]  = np.nan
    d_plc[mask_dam]  = np.nan
    d_runs[mask_dam] = np.nan
    shinba["dam_shinba_win"]   = d_win
    shinba["dam_shinba_place"] = d_plc
    shinba["dam_shinba_runs"]  = d_runs

    print("累積父父(祖父)統計（全コース）を計算中...")
    gs_win, gs_plc, _ = _build_cumulative_stats(shinba, ["grandsire"], "_is_win", "_is_plc")
    shinba["grandsire_win"]   = gs_win
    shinba["grandsire_place"] = gs_plc

    print("累積騎手統計を計算中...")
    j_win, j_plc, _ = _build_cumulative_stats(shinba, ["jockey_id"], "_is_win", "_is_plc")
    shinba["jockey_shinba_win"]   = j_win
    shinba["jockey_shinba_place"] = j_plc

    print("累積調教師統計を計算中...")
    t_win, t_plc, _ = _build_cumulative_stats(shinba, ["調教師"], "_is_win", "_is_plc")
    shinba["trainer_shinba_win"]   = t_win
    shinba["trainer_shinba_place"] = t_plc

    # 特徴量 DataFrame 組み立て
    feat = pd.DataFrame({
        "sire_course_win":     shinba["sire_course_win"],
        "sire_course_place":   shinba["sire_course_place"],
        "sire_course_runs":    shinba["sire_course_runs"],
        "sire_global_win":     shinba["sire_global_win"],
        "sire_global_place":   shinba["sire_global_place"],
        "bms_course_win":      shinba["bms_course_win"],
        "bms_course_place":    shinba["bms_course_place"],
        "bms_course_runs":     shinba["bms_course_runs"],
        "bms_global_win":      shinba["bms_global_win"],
        "bms_global_place":    shinba["bms_global_place"],
        "dam_shinba_win":      shinba["dam_shinba_win"],
        "dam_shinba_place":    shinba["dam_shinba_place"],
        "dam_shinba_runs":     shinba["dam_shinba_runs"],
        "grandsire_win":       shinba["grandsire_win"],
        "grandsire_place":     shinba["grandsire_place"],
        "cross_score":         shinba["cross_score"],
        "jockey_shinba_win":   shinba["jockey_shinba_win"],
        "jockey_shinba_place": shinba["jockey_shinba_place"],
        "trainer_shinba_win":  shinba["trainer_shinba_win"],
        "trainer_shinba_place":shinba["trainer_shinba_place"],
        "place_id":            shinba["place_id_int"].astype(float),
        "race_type_code":      shinba["race_type_code"],
        "course_len":          shinba["course_len_f"],
        "waku":                shinba["waku_int"],
        "umaban":              shinba["umaban_int"],
        "kinryo":              shinba["kinryo_f"],
        "sex_code":            shinba["sex_code"],
        "horse_weight":        shinba["horse_weight"],
        "birth_month":         shinba["birth_month"],
    })
    assert list(feat.columns) == SHINBA_FEATURES

    labels = shinba["label"].values
    race_ids = shinba["race_id"].values
    _, groups = np.unique(race_ids, return_counts=True)
    # LambdaRank は race_id ごとの連続グループが必要なので順番を保持
    groups = []
    cur_rid = None
    cur_cnt = 0
    for rid in race_ids:
        if rid != cur_rid:
            if cur_cnt > 0:
                groups.append(cur_cnt)
            cur_rid = rid
            cur_cnt = 1
        else:
            cur_cnt += 1
    if cur_cnt > 0:
        groups.append(cur_cnt)

    print(f"データセット完成: {len(feat)}行 / {len(groups)}レース")
    return feat, groups, labels


def build_stats_tables():
    """
    予測時に使う統計テーブルを全データから構築して返す。

    Returns:
        dict: {
            "sire_course":  DataFrame(sire, race_type, dist_band → win_rate, place_rate, runs),
            "sire_global":  DataFrame(sire → global_win, global_place),
            "bms_course":   DataFrame(bms, race_type, dist_band → win_rate, place_rate, runs),
            "bms_global":   DataFrame(bms → global_win, global_place),
            "dam":          DataFrame(dam → win_rate, place_rate, runs) ※runs<3除外済み,
            "grandsire":    DataFrame(grandsire → global_win, global_place),
            "jockey":       DataFrame(jockey_id → win_rate, place_rate),
            "trainer":      DataFrame(調教師 → win_rate, place_rate),
        }
    """
    shinba = _load_all_shinba_results()
    if shinba.empty:
        return {}

    unique_ids = shinba["horse_id"].dropna().unique().tolist()
    peds_map = _fetch_peds_map(unique_ids)
    shinba["sire"]      = shinba["horse_id"].map(lambda x: peds_map.get(str(x), ("","","",""))[0])
    shinba["dam"]       = shinba["horse_id"].map(lambda x: peds_map.get(str(x), ("","","",""))[1])
    shinba["grandsire"] = shinba["horse_id"].map(lambda x: peds_map.get(str(x), ("","","",""))[2])
    shinba["bms"]       = shinba["horse_id"].map(lambda x: peds_map.get(str(x), ("","","",""))[3])

    shinba["dist_band"] = shinba["course_len"].apply(
        lambda x: _dist_band(x) if str(x).isdigit() else "unknown"
    )

    def _rank_int(v):
        s = re.sub(r"\D", "", str(v))
        return int(s) if s else None

    shinba["_rank_i"] = shinba["着順"].apply(_rank_int)
    shinba["_is_win"] = (shinba["_rank_i"] == 1).astype(float)
    shinba["_is_plc"] = shinba["_rank_i"].apply(
        lambda r: 1.0 if r is not None and r <= 3 else 0.0
    )
    shinba = shinba.dropna(subset=["_rank_i"]).copy()

    sire_course = (
        shinba.groupby(["sire", "race_type", "dist_band"])
        .agg(win_rate=("_is_win", "mean"), place_rate=("_is_plc", "mean"), runs=("_is_win", "count"))
        .reset_index()
    )

    sire_global = (
        shinba.groupby("sire")
        .agg(global_win=("_is_win", "mean"), global_place=("_is_plc", "mean"))
        .reset_index()
    )

    bms_course = (
        shinba.groupby(["bms", "race_type", "dist_band"])
        .agg(win_rate=("_is_win", "mean"), place_rate=("_is_plc", "mean"), runs=("_is_win", "count"))
        .reset_index()
    )

    bms_global = (
        shinba.groupby("bms")
        .agg(global_win=("_is_win", "mean"), global_place=("_is_plc", "mean"))
        .reset_index()
    )

    dam_all = (
        shinba.groupby("dam")
        .agg(win_rate=("_is_win", "mean"), place_rate=("_is_plc", "mean"), runs=("_is_win", "count"))
        .reset_index()
    )
    # 出走数が少ない牝系は信頼性低いのでフィルタ
    dam_stats = dam_all[dam_all["runs"] >= _DAM_MIN_RUNS].copy()

    grandsire = (
        shinba.groupby("grandsire")
        .agg(global_win=("_is_win", "mean"), global_place=("_is_plc", "mean"))
        .reset_index()
    )

    jockey = (
        shinba.groupby("jockey_id")
        .agg(win_rate=("_is_win", "mean"), place_rate=("_is_plc", "mean"))
        .reset_index()
    )

    trainer = (
        shinba.groupby("調教師")
        .agg(win_rate=("_is_win", "mean"), place_rate=("_is_plc", "mean"))
        .reset_index()
        .rename(columns={"調教師": "trainer"})
    )

    return {
        "sire_course": sire_course,
        "sire_global": sire_global,
        "bms_course":  bms_course,
        "bms_global":  bms_global,
        "dam":         dam_stats,
        "grandsire":   grandsire,
        "jockey":      jockey,
        "trainer":     trainer,
    }


def make_row_for_prediction(
    sire, bms, dam, grandsire,
    race_type, course_len, place_id, waku, umaban, kinryo, sex_code,
    horse_weight, jockey_id, trainer_name,
    stats_tables,
    birth_month=None,
    cross_score=0.0,
):
    """
    1馬分の予測用特徴量行を返す（stats_tables は build_stats_tables() の戻り値）。

    Returns:
        pd.DataFrame: 1行 × SHINBA_FEATURES列
    """
    dist_band = _dist_band(course_len)
    rt = str(race_type)

    def _lookup(tbl, key_cols, key_vals, val_col, default=np.nan):
        if tbl is None or tbl.empty:
            return default
        mask = pd.Series([True] * len(tbl))
        for col, val in zip(key_cols, key_vals):
            mask &= tbl[col].astype(str) == str(val)
        rows = tbl[mask]
        return rows[val_col].iloc[0] if not rows.empty else default

    sc_tbl   = stats_tables.get("sire_course")
    sg_tbl   = stats_tables.get("sire_global")
    bc_tbl   = stats_tables.get("bms_course")
    bg_tbl   = stats_tables.get("bms_global")
    dam_tbl  = stats_tables.get("dam")
    gs_tbl   = stats_tables.get("grandsire")
    jk_tbl   = stats_tables.get("jockey")
    tr_tbl   = stats_tables.get("trainer")

    row = {
        "sire_course_win":     _lookup(sc_tbl,  ["sire", "race_type", "dist_band"], [sire, rt, dist_band], "win_rate"),
        "sire_course_place":   _lookup(sc_tbl,  ["sire", "race_type", "dist_band"], [sire, rt, dist_band], "place_rate"),
        "sire_course_runs":    _lookup(sc_tbl,  ["sire", "race_type", "dist_band"], [sire, rt, dist_band], "runs", 0.0),
        "sire_global_win":     _lookup(sg_tbl,  ["sire"], [sire], "global_win"),
        "sire_global_place":   _lookup(sg_tbl,  ["sire"], [sire], "global_place"),
        "bms_course_win":      _lookup(bc_tbl,  ["bms",  "race_type", "dist_band"], [bms,  rt, dist_band], "win_rate"),
        "bms_course_place":    _lookup(bc_tbl,  ["bms",  "race_type", "dist_band"], [bms,  rt, dist_band], "place_rate"),
        "bms_course_runs":     _lookup(bc_tbl,  ["bms",  "race_type", "dist_band"], [bms,  rt, dist_band], "runs", 0.0),
        "bms_global_win":      _lookup(bg_tbl,  ["bms"],  [bms],  "global_win"),
        "bms_global_place":    _lookup(bg_tbl,  ["bms"],  [bms],  "global_place"),
        "dam_shinba_win":      _lookup(dam_tbl, ["dam"],  [dam],  "win_rate"),
        "dam_shinba_place":    _lookup(dam_tbl, ["dam"],  [dam],  "place_rate"),
        "dam_shinba_runs":     _lookup(dam_tbl, ["dam"],  [dam],  "runs", 0.0),
        "grandsire_win":       _lookup(gs_tbl,  ["grandsire"], [grandsire], "global_win"),
        "grandsire_place":     _lookup(gs_tbl,  ["grandsire"], [grandsire], "global_place"),
        "cross_score":         float(cross_score) if cross_score is not None else 0.0,
        "jockey_shinba_win":   _lookup(jk_tbl,  ["jockey_id"], [jockey_id],   "win_rate"),
        "jockey_shinba_place": _lookup(jk_tbl,  ["jockey_id"], [jockey_id],   "place_rate"),
        "trainer_shinba_win":  _lookup(tr_tbl,  ["trainer"],   [trainer_name], "win_rate"),
        "trainer_shinba_place":_lookup(tr_tbl,  ["trainer"],   [trainer_name], "place_rate"),
        "place_id":            float(place_id),
        "race_type_code":      1.0 if rt == "ダート" else 0.0,
        "course_len":          float(course_len),
        "waku":                float(waku) if waku is not None else np.nan,
        "umaban":              float(umaban) if umaban is not None else np.nan,
        "kinryo":              float(kinryo) if kinryo is not None else np.nan,
        "sex_code":            float(sex_code),
        "horse_weight":        float(horse_weight) if horse_weight is not None else np.nan,
        "birth_month":         float(birth_month) if birth_month is not None else np.nan,
    }
    return pd.DataFrame([row])[SHINBA_FEATURES]
