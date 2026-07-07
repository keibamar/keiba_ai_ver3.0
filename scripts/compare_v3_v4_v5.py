"""
2026年 全場 v3blend / v4 / v5 的中率・回収率 一括集計

実行:
    python scripts/compare_v3_v4_v5.py

- v3blend : v3モデル(64特徴量) + race_result 最終オッズでブレンド
- v4      : v4モデル(67特徴量) の生スコアで順位付け
- v5      : v5モデル(70特徴量) の生スコアで順位付け（父/母父/父父の血統カテゴリカル追加）
            ※モデルが存在しないコースはそのレースをスキップ
"""
import sys, warnings, glob, itertools, os, time
warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import paths
from src.config.constants import PLACE_LIST
from src.logic.prediction.race_prediction_engine import (
    make_dataset_for_lightgbm_v3,
    _build_jockey_course_stats,
    get_lightgbm_v3_model,
    rank_index,
    prediction_error,
    _softmax,
)
from src.managers import race_card_dataset_manager
from src.PredictionModels.LightGBM.make_dataset_v5 import (
    build_pedigree_vocab, get_pedigree_cats,
)

BET_UNIT = 100
MODELS   = ["v3blend", "v4", "v5"]

# 血統vocab（起動時1回だけロード）
print("血統vocab読み込み中...")
_VOCAB = build_pedigree_vocab()
print(f"  {len(_VOCAB)}種類")


# ---------- モデルロード ----------

def _get_model(place_id, race_type, length, suffix):
    type_str = "turf" if race_type == "芝" else "dirt"
    mp = os.path.join(paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1],
                      f"{type_str}{length}_lambdarank_model{suffix}.txt")
    if not os.path.isfile(mp):
        raise FileNotFoundError(mp)
    return lgb.Booster(model_file=mp)


# ---------- 特徴量ビルド ----------

def build_v3_features(race_id, rc_df, race_info_df):
    try:
        place_id   = int(str(race_id)[4:6])
        race_type  = race_info_df.at[0, "race_type"]
        course_len = str(race_info_df.at[0, "course_len"])
        course_info = [place_id, race_type, course_len,
                       race_info_df.at[0, "ground_state"], race_info_df.at[0, "class"]]

        horse_ids  = rc_df["horse_id"].tolist()
        jockey_ids = rc_df["jockey_id"].tolist()
        waku_df    = rc_df[["枠", "馬番"]].reset_index(drop=True)

        dataset = pd.DataFrame()
        for hid in horse_ids:
            row = make_dataset_for_lightgbm_v3(race_id, course_info, hid)
            dataset = pd.concat([dataset.reset_index(drop=True), row.reset_index(drop=True)])

        js = _build_jockey_course_stats(place_id)
        wr_list, pr_list = [], []
        for jid in jockey_ids:
            wr, pr = js.get((str(jid), race_type, course_len), (np.nan, np.nan))
            wr_list.append(wr); pr_list.append(pr)

        full_df_64 = pd.concat([
            dataset.reset_index(drop=True),
            pd.DataFrame({"jockey_win_rate": wr_list, "jockey_place_rate": pr_list}),
            waku_df.reset_index(drop=True),
        ], axis=1).fillna(-1)

        model_v3  = get_lightgbm_v3_model(place_id, race_type, course_len)
        y_pred_v3 = model_v3.predict(full_df_64, num_iteration=model_v3.best_iteration)
        return full_df_64, y_pred_v3, place_id, race_type, course_len, True
    except Exception as e:
        prediction_error(e)
        return None, None, None, None, None, False


# ---------- 各モデルのランク計算 ----------

def v3blend_ranks(y_pred_v3, final_odds_series):
    p_ai = _softmax(np.array(y_pred_v3, dtype=float))
    try:
        odds_arr = final_odds_series.astype(float).values
        if np.all(np.isfinite(odds_arr)) and np.all(odds_arr > 0):
            inv = 1.0 / odds_arr
            p_combined = 0.5 * p_ai + 0.5 * (inv / inv.sum())
        else:
            p_combined = p_ai
    except Exception:
        p_combined = p_ai
    return rank_index(p_combined)


def v4_ranks(full_df_64, place_id, race_type, course_len, final_odds_series, final_pop_series):
    try:
        odds_arr = final_odds_series.astype(float).values
        pop_arr  = final_pop_series.astype(float).values
        extra = pd.DataFrame({
            "current_odds":       np.where(np.isfinite(odds_arr), odds_arr, -1),
            "current_popularity": np.where(np.isfinite(pop_arr),  pop_arr,  -1),
        })
        full_df_66 = pd.concat([full_df_64.reset_index(drop=True),
                                 extra.reset_index(drop=True)], axis=1)
        model  = _get_model(place_id, race_type, course_len, "_v4")
        y_pred = model.predict(full_df_66, num_iteration=model.best_iteration)
        return rank_index(y_pred)
    except Exception as e:
        prediction_error(e)
        return None


def v5_ranks(full_df_64, place_id, race_type, course_len,
             final_odds_series, final_pop_series, horse_ids):
    try:
        odds_arr = final_odds_series.astype(float).values
        pop_arr  = final_pop_series.astype(float).values
        father_ids, mf_ids, pgf_ids = [], [], []
        for hid in horse_ids:
            f_cat, mf_cat, pgf_cat = get_pedigree_cats(hid, _VOCAB)
            father_ids.append(f_cat)
            mf_ids.append(mf_cat)
            pgf_ids.append(pgf_cat)

        extra = pd.DataFrame({
            "current_odds":       np.where(np.isfinite(odds_arr), odds_arr, -1),
            "current_popularity": np.where(np.isfinite(pop_arr),  pop_arr,  -1),
            "father_cat":         father_ids,
            "mother_father_cat":  mf_ids,
            "paternal_gf_cat":    pgf_ids,
        })
        full_df_69 = pd.concat([full_df_64.reset_index(drop=True),
                                 extra.reset_index(drop=True)], axis=1)
        model  = _get_model(place_id, race_type, course_len, "_v5")
        y_pred = model.predict(full_df_69, num_iteration=model.best_iteration)
        return rank_index(y_pred)
    except Exception as e:
        prediction_error(e)
        return None


# ---------- 集計クラス ----------

class Stats:
    def __init__(self):
        self.n = 0
        self.tan_hit = self.tan_pay = self.tan_bet = 0
        self.fuku_hit = self.fuku_pay = self.fuku_bet = 0
        self.san_hit = self.san_pay = self.san_bet = 0

    def add(self, tan_h, tan_ret, fuku_h, fuku_ret, san_h, san_ret, san_bets):
        self.n += 1
        self.tan_bet  += BET_UNIT
        self.fuku_bet += BET_UNIT
        self.san_bet  += BET_UNIT * san_bets
        if tan_h:
            self.tan_hit += 1
            if tan_ret: self.tan_pay += tan_ret
        if fuku_h:
            self.fuku_hit += 1
            if fuku_ret: self.fuku_pay += fuku_ret
        if san_h:
            self.san_hit += 1
            if san_ret: self.san_pay += san_ret

    def pct(self, h): return f"{h}/{self.n}({100*h/self.n:.1f}%)" if self.n else "-/-"
    def rec(self, pay, bet): return f"{100*pay/bet:.1f}%" if bet else "-"

    def summary_line(self, label):
        return (
            f"  {label:<10} "
            f"単 {self.pct(self.tan_hit):>14} 回{self.rec(self.tan_pay,self.tan_bet):>6}  "
            f"複 {self.pct(self.fuku_hit):>14} 回{self.rec(self.fuku_pay,self.fuku_bet):>6}  "
            f"3連 {self.pct(self.san_hit):>14} 回{self.rec(self.san_pay,self.san_bet):>6}"
        )


def get_return(ret_df, shikibetsu, umaban_str=None):
    rows = ret_df[ret_df["式別"] == shikibetsu]
    if rows.empty: return None
    if umaban_str is not None:
        rows = rows[rows["馬番"].astype(str) == umaban_str]
        if rows.empty: return None
    return int(rows["配当"].iloc[0])


# ---------- メイン ----------

all_stats   = {m: Stats() for m in MODELS}
place_stats = {}
processed = skipped = 0
t0 = time.time()

active_places = []
for pid in range(1, 11):
    pfx = f"2026{pid:02d}"
    cnt = sum(len(glob.glob(f"{d}/{pfx}*.csv")) for d in glob.glob("data/race_card/2026*"))
    if cnt > 0:
        active_places.append(pid)

print("=" * 90)
print("2026年 的中率・回収率集計  全場 v3blend / v4 / v5  (100円/点)")
print(f"対象場: {[PLACE_LIST[p-1] for p in active_places]}")
print("=" * 90)

for place_id in active_places:
    place_name = PLACE_LIST[place_id - 1]
    place_stats[place_name] = {m: Stats() for m in MODELS}

    prefix = f"2026{place_id:02d}"
    race_paths = []
    for dd in sorted(glob.glob("data/race_card/2026*")):
        race_paths += sorted(glob.glob(f"{dd}/{prefix}*.csv"))

    print(f"\n[{place_name}] {len(race_paths)}R 処理中...")

    for cp in race_paths:
        race_id = os.path.basename(cp).replace(".csv", "")

        res_paths = glob.glob(f"data/race_result/**/{race_id}.csv", recursive=True)
        if not res_paths: skipped += 1; continue
        res_df = pd.read_csv(res_paths[0], index_col=0)
        if "着順" not in res_df.columns: skipped += 1; continue

        ret_paths = glob.glob(f"data/race_info/race_returns/**/{race_id}.csv", recursive=True)
        if not ret_paths: skipped += 1; continue
        ret_df = pd.read_csv(ret_paths[0], index_col=0)

        rc_df = pd.read_csv(cp, index_col=0).reset_index(drop=True)
        if "枠" not in rc_df.columns or "horse_id" not in rc_df.columns:
            skipped += 1; continue

        race_info_df = race_card_dataset_manager.get_race_info_csv(race_id)
        if race_info_df.empty: skipped += 1; continue

        nums       = pd.to_numeric(res_df["着順"], errors="coerce")
        winner_set = set(res_df[nums == 1]["馬番"].astype(str))
        top3_set   = set(res_df[nums <= 3]["馬番"].astype(str))

        san_row    = ret_df[ret_df["式別"] == "三連複"]
        san_winner = set(san_row["馬番"].iloc[0].split("-")) if not san_row.empty else None
        san_odds   = int(san_row["配当"].iloc[0]) if not san_row.empty else None

        key_series = rc_df["馬番"].astype(str).str.strip()
        odds_map   = dict(zip(res_df["馬番"].astype(str), pd.to_numeric(res_df["単勝"], errors="coerce")))
        pop_map    = dict(zip(res_df["馬番"].astype(str), pd.to_numeric(res_df["人気"], errors="coerce")))
        final_odds = key_series.map(odds_map)
        final_pop  = key_series.map(pop_map)

        full_df_64, y_pred_v3, pid, rtype, clen, ok = build_v3_features(
            race_id, rc_df, race_info_df)

        v3_top5 = []
        if ok:
            ranks     = v3blend_ranks(y_pred_v3, final_odds)
            v3_sorted = np.argsort(ranks)
            v3_top5   = [str(int(float(rc_df["馬番"].iloc[i]))) for i in v3_sorted[:5]]

        v4_top5 = []
        if ok:
            ranks_v4 = v4_ranks(full_df_64, pid, rtype, clen, final_odds, final_pop)
            if ranks_v4 is not None:
                v4_sorted = np.argsort(ranks_v4)
                v4_top5   = [str(int(float(rc_df["馬番"].iloc[i]))) for i in v4_sorted[:5]]

        v5_top5 = []
        if ok:
            horse_ids_list = rc_df["horse_id"].tolist()
            ranks_v5 = v5_ranks(full_df_64, pid, rtype, clen, final_odds, final_pop, horse_ids_list)
            if ranks_v5 is not None:
                v5_sorted = np.argsort(ranks_v5)
                v5_top5   = [str(int(float(rc_df["馬番"].iloc[i]))) for i in v5_sorted[:5]]

        for mname, top5 in [("v3blend", v3_top5), ("v4", v4_top5), ("v5", v5_top5)]:
            if not top5: continue
            honmei   = top5[0]
            tan_h    = honmei in winner_set
            tan_ret  = get_return(ret_df, "単勝")
            fuku_h   = honmei in top3_set
            fuku_ret = get_return(ret_df, "複勝", honmei)
            san_combs = list(itertools.combinations(top5[:5], 3))
            san_h = (san_winner is not None) and any(
                {str(int(float(a))), str(int(float(b))), str(int(float(c)))} == san_winner
                for a, b, c in san_combs
            )
            all_stats[mname].add(tan_h, tan_ret, fuku_h, fuku_ret, san_h, san_odds, len(san_combs))
            place_stats[place_name][mname].add(tan_h, tan_ret, fuku_h, fuku_ret, san_h, san_odds, len(san_combs))

        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {processed}R完了  ({elapsed:.0f}秒)")

    print(f"  --- {place_name} 小計 ({place_stats[place_name]['v3blend'].n}R) ---")
    for m in MODELS:
        s = place_stats[place_name][m]
        if s.n > 0:
            print(s.summary_line(m))

# ===== 総合 =====
elapsed_total = time.time() - t0
print(f"\n{'='*90}")
print(f"全場合計: {processed}R処理  スキップ:{skipped}R  所要:{elapsed_total:.0f}秒")
print("-" * 90)
print(f"{'モデル':<10}  {'単勝的中率':^18} {'回収率':^8}  {'複勝的中率':^18} {'回収率':^8}  {'3連複5頭的中率':^18} {'回収率':^8}  R数")
print("-" * 90)
for m in MODELS:
    s = all_stats[m]
    if s.n == 0: continue
    print(f"{m:<10}  "
          f"{s.pct(s.tan_hit):^18} {s.rec(s.tan_pay,s.tan_bet):^8}  "
          f"{s.pct(s.fuku_hit):^18} {s.rec(s.fuku_pay,s.fuku_bet):^8}  "
          f"{s.pct(s.san_hit):^18} {s.rec(s.san_pay,s.san_bet):^8}  {s.n}")
print("=" * 90)
