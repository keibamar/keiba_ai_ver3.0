"""eval_nodds_no26.py

オッズ不使用モデル（v11_nodds / v12_nodds / v15_nodds）の 2026年評価スクリプト。
train_nodds_no26.py で学習したモデルを使用。

特徴量: current_odds / current_popularity を除外した 88/95/102 列。
キャッシュ: logs/race_records_nodds_2026.pkl

実行: python scripts/eval_nodds_no26.py
"""

import os, sys, glob, time, pickle, warnings, re
warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")

import numpy as np
import pandas as pd
import lightgbm as lgb
import itertools

from src.config import paths
from src.config.constants import PLACE_LIST
from src.logic.prediction.race_prediction_engine import (
    make_dataset_for_lightgbm_v3,
    _build_jockey_course_stats,
    prediction_error,
)
from src.managers import race_card_dataset_manager
from src.PredictionModels.LightGBM.make_dataset_v5 import build_pedigree_vocab, get_pedigree_cats
from src.PredictionModels.LightGBM.make_dataset_v6 import get_extra_past_race_features_v6
from src.PredictionModels.LightGBM.make_dataset_v2 import _parse_weight
from src.PredictionModels.LightGBM.make_dataset_v7 import _parse_kinryo
from src.PredictionModels.LightGBM.make_dataset_v8 import get_extra_past_race_features_v8
from src.PredictionModels.LightGBM.make_dataset_v9 import (
    get_extra_past_race_features_v9, index_v9,
)
import past_performance

BET_UNIT  = 100
EVAL_YEAR = 2026
NODDS_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_nodds_{EVAL_YEAR}.pkl")

# ── 列定義 ──
_EXCLUDE = {"race_id", "current_odds", "current_popularity"}

# feat_rows_nodds は 102要素（index_v9 から _EXCLUDE を除いた重複ありリスト）
_NODDS_COLS_WITH_DUPS = [c for c in index_v9 if c not in _EXCLUDE]  # 102列

# スコアリング用: 重複排除後 100列
_seen = set()
NODDS_COLS = []
for _c in _NODDS_COLS_WITH_DUPS:
    if _c not in _seen:
        NODDS_COLS.append(_c)
        _seen.add(_c)
# 重複排除後 100列

_v7_end = NODDS_COLS.index("horse_weight_abs_1") + 1   # v11_nodds 境界 (86)
_v8_end = NODDS_COLS.index("time_diff_trend_5")  + 1   # v12_nodds 境界 (93)
V11_NODDS_COLS = NODDS_COLS[:_v7_end]
V12_NODDS_COLS = NODDS_COLS[:_v8_end]
V15_NODDS_COLS = NODDS_COLS   # 100列

print("血統vocab読み込み中...")
_VOCAB = build_pedigree_vocab()
print(f"  {len(_VOCAB)}種類")

_lgb_cache = {}


def _get_lgb(place_id, race_type, length, suffix):
    key = (place_id, race_type, length, suffix)
    if key in _lgb_cache:
        return _lgb_cache[key]
    type_str = "turf" if race_type == "芝" else "dirt"
    mp = os.path.join(paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1],
                      f"{type_str}{length}_lambdarank_model{suffix}.txt")
    if not os.path.isfile(mp):
        raise FileNotFoundError(mp)
    m = lgb.Booster(model_file=mp)
    _lgb_cache[key] = m
    return m


def _safe_float(v):
    try: return float(v)
    except: return np.nan


def _current_date(race_card_path):
    try:
        return pd.to_datetime(os.path.basename(os.path.dirname(race_card_path)), format="%Y%m%d")
    except:
        return pd.NaT


def build_features(race_id, rc_df, race_info_df, race_card_path):
    """オッズ不使用版: full_df_64, ped_cats, v6_df, v7_df, v8_df, feat_rows_nodds を構築。"""
    try:
        place_id   = int(str(race_id)[4:6])
        race_type  = race_info_df.at[0, "race_type"]
        course_len = str(race_info_df.at[0, "course_len"])
        course_info = [place_id, race_type, course_len,
                       race_info_df.at[0, "ground_state"], race_info_df.at[0, "class"]]

        horse_ids  = rc_df["horse_id"].tolist()
        jockey_ids = rc_df["jockey_id"].tolist()
        waku_df    = rc_df[["枠", "馬番"]].reset_index(drop=True)
        kinryo_col = rc_df.columns[4]
        n_horses   = float(len(rc_df))
        current_dt = _current_date(race_card_path)

        js = _build_jockey_course_stats(place_id)

        dataset_v3 = pd.DataFrame()
        extra_v6_rows, extra_v8_rows, extra_v9_rows = [], [], []
        ped_cats = []
        days_since_list, n_horses_1_list, hw_abs_1_list = [], [], []
        feat_rows_nodds = []

        for idx, hid in enumerate(horse_ids):
            # v3 base (60列)
            try:
                row_v3 = make_dataset_for_lightgbm_v3(race_id, course_info, hid)
                v3_vals = row_v3.values[0].tolist() if not row_v3.empty else [np.nan] * 60
            except:
                v3_vals = [np.nan] * 60
            dataset_v3 = pd.concat([dataset_v3.reset_index(drop=True),
                                    pd.DataFrame([v3_vals])])

            # pedigree cats
            f_cat, mf_cat, pgf_cat = get_pedigree_cats(hid, _VOCAB)
            ped_cats.append((f_cat, mf_cat, pgf_cat))

            # past 5 races
            try:
                r5 = past_performance.get_past_race_info(hid, race_id, race_num=5)
            except:
                r5 = pd.DataFrame()

            extra_v6 = get_extra_past_race_features_v6(r5) if not r5.empty else [np.nan] * 16
            extra_v8 = get_extra_past_race_features_v8(r5) if not r5.empty else [np.nan] * 7
            extra_v9 = get_extra_past_race_features_v9(r5) if not r5.empty else [np.nan] * 7
            extra_v6_rows.append(extra_v6)
            extra_v8_rows.append(extra_v8)
            extra_v9_rows.append(extra_v9)

            # v7 extras
            days_since = n_horses_1 = hw_abs_1 = np.nan
            if not r5.empty:
                if not pd.isna(current_dt):
                    try:
                        prev_dt = pd.to_datetime(
                            str(r5.iloc[0].get("日付", "")).strip().replace("/", "-"),
                            format="%Y-%m-%d"
                        )
                        days_since = float((current_dt - prev_dt).days)
                    except: pass
                n_horses_1 = _safe_float(r5.iloc[0].get("頭数", ""))
                hw_abs_1   = _parse_weight(r5.iloc[0].get("馬体重", ""))
            days_since_list.append(days_since)
            n_horses_1_list.append(n_horses_1)
            hw_abs_1_list.append(hw_abs_1)

            # nodds feat row (102列、race_id・current_odds・current_popularity 除く)
            jid = str(jockey_ids[idx])
            wr, pr = js.get((jid, race_type, course_len), (np.nan, np.nan))
            kinryo  = _parse_kinryo(rc_df[kinryo_col].iloc[idx])
            waku    = _safe_float(waku_df.at[idx, "枠"])
            umaban  = _safe_float(waku_df.at[idx, "馬番"])

            row_nodds = (
                v3_vals[0:60]
                + [wr, pr]                      # jockey (2)
                + [waku, umaban]                # waku/umaban (2)
                # current_odds / current_popularity を入れない
                + [f_cat, mf_cat, pgf_cat]      # ped_cats (3)
                + list(extra_v6)                # v6_extra (16)
                + [kinryo, days_since, n_horses, n_horses_1, hw_abs_1]  # v7_extra (5)
                + list(extra_v8[:5])            # corner_chase (5)
                + [extra_v8[5] if len(extra_v8) > 5 else np.nan,
                   extra_v8[6] if len(extra_v8) > 6 else np.nan]       # agari_trend, time_diff_trend (2)
                + list(extra_v9[:5])            # agari_df_course (5)
                + [extra_v9[5] if len(extra_v9) > 5 else np.nan,
                   extra_v9[6] if len(extra_v9) > 6 else np.nan]       # corner_ratio_std5, agari_std5 (2)
            )                                   # 合計 102列
            feat_rows_nodds.append(row_nodds)

        # DataFrame化
        dataset_v3 = dataset_v3.reset_index(drop=True)
        v6_df = pd.DataFrame(extra_v6_rows)
        v8_df = pd.DataFrame(extra_v8_rows, columns=[
            "corner_chase_1","corner_chase_2","corner_chase_3",
            "corner_chase_4","corner_chase_5","agari_trend_5","time_diff_trend_5",
        ])
        wr_list = [js.get((str(j), race_type, course_len), (np.nan, np.nan))[0] for j in jockey_ids]
        pr_list = [js.get((str(j), race_type, course_len), (np.nan, np.nan))[1] for j in jockey_ids]
        v7_df = pd.DataFrame({
            "kinryo":              [_parse_kinryo(rc_df[kinryo_col].iloc[i]) for i in range(len(horse_ids))],
            "days_since_last_race": days_since_list,
            "n_horses_today":       [n_horses] * len(horse_ids),
            "n_horses_1":           n_horses_1_list,
            "horse_weight_abs_1":   hw_abs_1_list,
        })

        full_df_64 = pd.concat([
            dataset_v3,
            pd.DataFrame({"jockey_win_rate": wr_list, "jockey_place_rate": pr_list}),
            waku_df,
        ], axis=1).fillna(-1)

        return (full_df_64, ped_cats, v6_df, v7_df, v8_df, feat_rows_nodds,
                place_id, race_type, course_len, horse_ids, True)
    except Exception as e:
        prediction_error(e)
        return None, None, None, None, None, None, None, None, None, None, False


def _score_nodds(feat_rows_nodds, pid, rtype, clen, suffix, cols):
    """feat_rows_nodds（102列）から cols 列だけ抽出してスコアリング。"""
    try:
        df = pd.DataFrame(feat_rows_nodds, columns=_NODDS_COLS_WITH_DUPS).fillna(-1)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]  # 100列に重複排除
        df_sub = df[cols]
        m = _get_lgb(pid, rtype, clen, suffix)
        return m.predict(df_sub, num_iteration=m.best_iteration)
    except Exception as e:
        prediction_error(e)
        return None


def get_return(ret_df, shikibetsu, umaban_str=None):
    rows = ret_df[ret_df["式別"] == shikibetsu]
    if rows.empty: return None
    if umaban_str is not None:
        rows = rows[rows["馬番"].astype(str) == umaban_str]
        if rows.empty: return None
    return int(rows["配当"].iloc[0])


# ── キャッシュ確認 ──
if os.path.exists(NODDS_CACHE):
    print(f"\nキャッシュ読み込み: {NODDS_CACHE}")
    with open(NODDS_CACHE, "rb") as f:
        race_records = pickle.load(f)
    print(f"  {len(race_records)}R")
else:
    active_places = []
    for pid in range(1, 11):
        pfx = f"{EVAL_YEAR}{pid:02d}"
        cnt = sum(len(glob.glob(f"{d}/{pfx}*.csv")) for d in glob.glob(f"data/race_card/{EVAL_YEAR}*"))
        if cnt > 0:
            active_places.append(pid)
    print(f"対象場: {[PLACE_LIST[p-1] for p in active_places]}")

    race_records = []
    t0 = time.time()
    processed = skipped = 0

    for place_id in active_places:
        prefix = f"{EVAL_YEAR}{place_id:02d}"
        race_paths = []
        for dd in sorted(glob.glob(f"data/race_card/{EVAL_YEAR}*")):
            race_paths += sorted(glob.glob(f"{dd}/{prefix}*.csv"))

        for cp in race_paths:
            race_id = os.path.basename(cp).replace(".csv", "")
            res_paths = glob.glob(f"data/race_result/**/{race_id}.csv", recursive=True)
            if not res_paths: skipped += 1; continue
            res_df = pd.read_csv(res_paths[0], index_col=0)
            if "着順" not in res_df.columns: skipped += 1; continue
            ret_paths = glob.glob(f"data/race_info/race_returns/**/{race_id}.csv", recursive=True)
            if not ret_paths: skipped += 1; continue
            ret_df = pd.read_csv(ret_paths[0], index_col=0)
            rc_df  = pd.read_csv(cp, index_col=0).reset_index(drop=True)
            if "枠" not in rc_df.columns or "horse_id" not in rc_df.columns: skipped += 1; continue
            race_info_df = race_card_dataset_manager.get_race_info_csv(race_id)
            if race_info_df.empty: skipped += 1; continue

            nums       = pd.to_numeric(res_df["着順"], errors="coerce")
            winner_set = set(res_df[nums == 1]["馬番"].astype(str))
            top3_set   = set(res_df[nums <= 3]["馬番"].astype(str))
            san_row    = ret_df[ret_df["式別"] == "三連複"]
            san_winner = set(san_row["馬番"].iloc[0].split("-")) if not san_row.empty else None
            san_odds   = int(san_row["配当"].iloc[0]) if not san_row.empty else None

            umabans    = [str(int(float(u))) for u in rc_df["馬番"].tolist()]
            tan_ret    = get_return(ret_df, "単勝")
            fuku_rets  = {u: get_return(ret_df, "複勝", u) for u in umabans}

            result = build_features(race_id, rc_df, race_info_df, cp)
            (full_df_64, ped_cats, v6_df, v7_df, v8_df, feat_rows_nodds,
             pid, rtype, clen, horse_ids, ok) = result
            if not ok: skipped += 1; continue

            s_v11n = _score_nodds(feat_rows_nodds, pid, rtype, clen, "_v11nodds_no26", V11_NODDS_COLS)
            s_v12n = _score_nodds(feat_rows_nodds, pid, rtype, clen, "_v12nodds_no26", V12_NODDS_COLS)
            s_v15n = _score_nodds(feat_rows_nodds, pid, rtype, clen, "_v15nodds_no26", V15_NODDS_COLS)

            if s_v15n is None and s_v11n is None: skipped += 1; continue

            race_records.append({
                "race_id":    race_id,
                "winner_set": winner_set,
                "top3_set":   top3_set,
                "san_winner": san_winner,
                "san_odds":   san_odds,
                "tan_ret":    tan_ret,
                "fuku_rets":  fuku_rets,
                "umabans":    umabans,
                "s_v11n": np.array(s_v11n, dtype=float) if s_v11n is not None else None,
                "s_v12n": np.array(s_v12n, dtype=float) if s_v12n is not None else None,
                "s_v15n": np.array(s_v15n, dtype=float) if s_v15n is not None else None,
            })
            processed += 1
            if processed % 200 == 0:
                print(f"  {processed}R収集済み ({time.time()-t0:.0f}秒)")
                with open(NODDS_CACHE, "wb") as _f:
                    pickle.dump(race_records, _f)

    print(f"\n収集完了: {processed}R ({time.time()-t0:.0f}秒), スキップ:{skipped}R")
    with open(NODDS_CACHE, "wb") as f:
        pickle.dump(race_records, f)
    print(f"キャッシュ保存: {NODDS_CACHE}")

# ── 評価 ──
def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def evaluate(records, key, label):
    n = tan_hit = tan_pay = tan_bet = 0
    fuku_hit = fuku_pay = fuku_bet = 0
    san_hit  = san_pay  = san_bet  = 0

    for r in records:
        s = r.get(key)
        if s is None: continue
        umabans = r["umabans"]
        honmei  = umabans[np.argmax(s)]

        # 3連複5頭: honmei + 指数上位4頭
        order = np.argsort(-s)
        san5 = [honmei]
        for i in order:
            u = umabans[i]
            if u != honmei: san5.append(u)
            if len(san5) == 5: break

        tan_bet  += BET_UNIT
        if honmei in r["winner_set"]:
            tan_hit += 1
            if r["tan_ret"]: tan_pay += r["tan_ret"]

        fuku_bet += BET_UNIT
        if honmei in r["top3_set"]:
            fuku_hit += 1
            ret = r["fuku_rets"].get(honmei)
            if ret: fuku_pay += ret

        combs = list(itertools.combinations(san5, 3))
        san_bet += BET_UNIT * len(combs)
        if r["san_winner"] is not None:
            if any({a, b, c} == r["san_winner"] for a, b, c in combs):
                san_hit += 1
                if r["san_odds"]: san_pay += r["san_odds"]

        n += 1

    if n == 0: return
    print(f"\n  【{label}】  n={n}R")
    print(f"    単勝: {100*tan_hit/n:.1f}% / {100*tan_pay/tan_bet:.1f}%")
    print(f"    複勝: {100*fuku_hit/n:.1f}% / {100*fuku_pay/fuku_bet:.1f}%")
    print(f"    3連複: {100*san_hit/n:.1f}% / {100*san_pay/san_bet:.1f}%  ({n*10}点×100円)")


print(f"\n{'='*70}")
print(f"  オッズ不使用モデル評価  {EVAL_YEAR}年  n={len(race_records)}R")
print(f"{'='*70}")

for key, label in [
    ("s_v11n", "v11_nodds (88列)"),
    ("s_v12n", "v12_nodds (95列)"),
    ("s_v15n", "v15_nodds (102列)"),
]:
    valid = [r for r in race_records if r.get(key) is not None]
    if valid:
        evaluate(valid, key, label)
    else:
        print(f"\n  【{label}】 モデル未学習（スキップ）")

# α スイープ: v15_nodds 単体の0.1刻み確認（参考）
print(f"\n{'='*70}")
print(f"  v15_nodds 単体 各指標確認完了")
print(f"{'='*70}")
