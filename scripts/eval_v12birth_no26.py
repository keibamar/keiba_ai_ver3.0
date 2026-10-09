"""eval_v12birth_no26.py

v12birth_no26（binary）と v12birth_ev_no26（EV回帰）を2026年データで評価。
v11ev ベースラインと的中率・回収率を比較する。

実行:
    python scripts/eval_v12birth_no26.py
"""

import glob, gc, itertools, os, pickle, sys, time, traceback, warnings
warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")

import lightgbm as lgb
import numpy as np
import pandas as pd

import name_header
from src.config import paths
from src.config.constants import PLACE_LIST
from src.PredictionModels.LightGBM.make_dataset_v12 import (
    make_dataset_for_train_v12, load_dataset_v12,
)

BET_UNIT  = 100
EVAL_YEAR = 2026
V12_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_v12birth_{EVAL_YEAR}.pkl")
_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "prediction", "datasets")


# ============================================================
# モデルロード
# ============================================================

def _get_model(place_id, race_type, length, suffix):
    type_str = "turf" if race_type == "芝" else "dirt"
    mp = os.path.join(paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1],
                      f"{type_str}{length}_lambdarank_model{suffix}.txt")
    if not os.path.isfile(mp):
        raise FileNotFoundError(mp)
    return lgb.Booster(model_file=mp)


# ============================================================
# 2026年 v12 データセット生成（未作成コースのみ）
# ============================================================

def _missing_courses_2026(place_id):
    out_dir = os.path.join(_DATASET_DIR, name_header.PLACE_LIST[place_id - 1])
    missing = []
    for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
        v12_path = os.path.join(out_dir, f"{EVAL_YEAR}_{race_type}{length}_ai_dataset_for_rank_v12.csv")
        if not os.path.isfile(v12_path):
            missing.append((race_type, length))
    return missing


def generate_2026_datasets():
    print(f"\n[Step 1] {EVAL_YEAR}年 v12 データセット生成（未作成コースのみ）")
    for place_id in range(1, 11):
        missing = _missing_courses_2026(place_id)
        if not missing:
            continue
        pname = name_header.PLACE_LIST[place_id - 1]
        print(f"  {pname}: {len(missing)} コース")
        try:
            make_dataset_for_train_v12(place_id, EVAL_YEAR, course_filter=missing)
        except Exception:
            print(f"    ERROR: {pname}")
            traceback.print_exc()
        gc.collect()


# ============================================================
# 払戻・結果データ取得
# ============================================================

def _get_return_df(race_id):
    paths_list = glob.glob(
        os.path.join(PROJECT_ROOT, "data", "race_info", "race_returns", "**", f"{race_id}.csv"),
        recursive=True
    )
    if not paths_list:
        return pd.DataFrame()
    return pd.read_csv(paths_list[0], index_col=0)


def _get_return(ret_df, shikibetsu, umaban_str=None):
    rows = ret_df[ret_df["式別"] == shikibetsu]
    if rows.empty:
        return None
    if umaban_str is not None:
        rows = rows[rows["馬番"].astype(str) == umaban_str]
        if rows.empty:
            return None
    return int(rows["配当"].iloc[0])


def _get_result_df(race_id):
    paths_list = glob.glob(
        os.path.join(PROJECT_ROOT, "data", "race_result", "**", f"{race_id}.csv"),
        recursive=True
    )
    if not paths_list:
        return pd.DataFrame()
    return pd.read_csv(paths_list[0], index_col=0)


# ============================================================
# v12 スコア収集
# ============================================================

def collect_v12_records():
    print(f"\n[Step 2] v12birth スコア収集 ({EVAL_YEAR}年)")
    race_records = []
    t0 = time.time()
    processed = skipped = 0

    for place_id in range(1, 11):
        pname = name_header.PLACE_LIST[place_id - 1]
        print(f"\n  [{pname}]")

        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            df, flag = load_dataset_v12(place_id, EVAL_YEAR, race_type, length)
            if df.empty or flag.empty:
                continue

            # v12birth（binary）モデル
            try:
                model_hit = _get_model(place_id, race_type, length, "_v12birth_no26")
            except FileNotFoundError:
                model_hit = None

            # v12birth_ev（EV回帰）モデル
            try:
                model_ev = _get_model(place_id, race_type, length, "_v12birth_ev_no26")
            except FileNotFoundError:
                model_ev = None

            # v11ev ベースライン
            try:
                model_v11ev = _get_model(place_id, race_type, length, "_v11ev_no26")
                # v11ev は v11 特徴量（birth_monthなし）を使う
                feat_cols_v11 = [c for c in df.columns if c not in ("race_id", "birth_month")]
                X_v11 = df[feat_cols_v11].fillna(-1)
                scores_v11ev = model_v11ev.predict(X_v11)
            except Exception:
                scores_v11ev = None

            feat_cols = [c for c in df.columns if c != "race_id"]
            X = df[feat_cols].fillna(-1)

            scores_hit = model_hit.predict(X) if model_hit else None
            scores_ev  = model_ev.predict(X)  if model_ev  else None

            df = df.copy().reset_index(drop=True)
            flag = flag.reset_index(drop=True)
            df["_flag"] = flag["result_flag"].values

            for rid, grp_idx in df.groupby("race_id").groups.items():
                race_id_str = str(int(rid))
                grp = df.loc[grp_idx].reset_index(drop=True)
                local_idx = grp_idx.tolist()

                ret_df = _get_return_df(race_id_str)
                res_df = _get_result_df(race_id_str)
                if ret_df.empty or res_df.empty:
                    skipped += 1
                    continue
                if "着順" not in res_df.columns or len(grp) != len(res_df):
                    skipped += 1
                    continue

                nums = pd.to_numeric(res_df["着順"], errors="coerce")
                winner_set = set(res_df[nums == 1]["馬番"].astype(str))
                top3_set   = set(res_df[nums <= 3]["馬番"].astype(str))
                san_row    = ret_df[ret_df["式別"] == "三連複"]
                san_winner = set(san_row["馬番"].iloc[0].split("-")) if not san_row.empty else None
                san_odds   = int(san_row["配当"].iloc[0]) if not san_row.empty else None

                umabans = [str(int(float(u))) for u in res_df["馬番"].tolist()]
                fuku_rets = {u: _get_return(ret_df, "複勝", u) for u in umabans}
                tan_ret   = _get_return(ret_df, "単勝")

                # シャッフルでタイブレーキング回避
                rng = np.random.default_rng(seed=int(rid))
                perm = rng.permutation(len(grp))
                umb_shuffled = [umabans[i] for i in perm]

                rec = {
                    "race_id":    race_id_str,
                    "place_id":   place_id,
                    "race_type":  race_type,
                    "length":     length,
                    "winner_set": winner_set,
                    "top3_set":   top3_set,
                    "san_winner": san_winner,
                    "san_odds":   san_odds,
                    "tan_ret":    tan_ret,
                    "fuku_rets":  fuku_rets,
                    "umabans":    umb_shuffled,
                }
                if scores_hit is not None:
                    s = scores_hit[local_idx][perm]
                    rec["s_v12birth"] = s
                if scores_ev is not None:
                    s = scores_ev[local_idx][perm]
                    rec["s_v12birth_ev"] = s
                if scores_v11ev is not None:
                    s = scores_v11ev[local_idx][perm]
                    rec["s_v11ev"] = s

                race_records.append(rec)
                processed += 1

        if processed % 100 == 0 and processed > 0:
            print(f"    {processed}R収集済み ({time.time()-t0:.0f}秒)")

    print(f"\n  収集完了: {processed}R / スキップ: {skipped}R ({time.time()-t0:.0f}秒)")
    return race_records


# ============================================================
# 評価
# ============================================================

class Stats:
    def __init__(self):
        self.n = self.tan_hit = self.tan_pay = self.tan_bet = 0
        self.fuku_hit = self.fuku_pay = self.fuku_bet = 0
        self.san_hit  = self.san_pay  = self.san_bet  = 0

    def add(self, tan_h, tan_ret, fuku_h, fuku_ret, san_h, san_ret, san_bets):
        self.n += 1
        self.tan_bet += BET_UNIT; self.fuku_bet += BET_UNIT
        self.san_bet += BET_UNIT * san_bets
        if tan_h:
            self.tan_hit += 1
            if tan_ret: self.tan_pay += tan_ret
        if fuku_h:
            self.fuku_hit += 1
            if fuku_ret: self.fuku_pay += fuku_ret
        if san_h:
            self.san_hit += 1
            if san_ret: self.san_pay += san_ret

    def tan_pct(self):  return 100 * self.tan_hit  / self.n if self.n else 0
    def fuku_pct(self): return 100 * self.fuku_hit / self.n if self.n else 0
    def san_pct(self):  return 100 * self.san_hit  / self.n if self.n else 0
    def tan_rec(self):  return 100 * self.tan_pay  / self.tan_bet  if self.tan_bet  else 0
    def fuku_rec(self): return 100 * self.fuku_pay / self.fuku_bet if self.fuku_bet else 0
    def san_rec(self):  return 100 * self.san_pay  / self.san_bet  if self.san_bet  else 0


def evaluate(label, records, score_key, width=60):
    st = Stats()
    for r in records:
        scores = r.get(score_key)
        if scores is None or len(scores) == 0:
            continue
        order    = np.argsort(-scores)
        umabans  = r["umabans"]
        top5     = [umabans[i] for i in order[:5] if i < len(umabans)]
        if not top5:
            continue
        honmei   = top5[0]
        san_combs = list(itertools.combinations(top5, 3))
        san_h = (r["san_winner"] is not None) and any(
            {a, b, c} == r["san_winner"] for a, b, c in san_combs
        )
        st.add(
            honmei in r["winner_set"], r["tan_ret"],
            honmei in r["top3_set"],   r["fuku_rets"].get(honmei),
            san_h, r["san_odds"], len(san_combs),
        )
    print(f"  {label:<{width}}  n={st.n:>4}  "
          f"単勝 {st.tan_pct():>5.1f}%/{st.tan_rec():>6.1f}%  "
          f"複勝 {st.fuku_pct():>5.1f}%/{st.fuku_rec():>6.1f}%  "
          f"3連複 {st.san_pct():>5.1f}%/{st.san_rec():>6.1f}%")
    return st


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"eval_v12birth_no26  評価年: {EVAL_YEAR}")
    print("v11ev ベースライン vs v12birth (binary) vs v12birth_ev (EV回帰)")
    print("=" * 70)

    # 2026年 v12 データセット生成
    generate_2026_datasets()

    # スコア収集（キャッシュ優先）
    if os.path.exists(V12_CACHE):
        print(f"\nキャッシュ読み込み: {V12_CACHE}")
        with open(V12_CACHE, "rb") as f:
            race_records = pickle.load(f)
        print(f"  {len(race_records)}R")
    else:
        race_records = collect_v12_records()
        os.makedirs(os.path.dirname(V12_CACHE), exist_ok=True)
        with open(V12_CACHE, "wb") as f:
            pickle.dump(race_records, f)
        print(f"  キャッシュ保存: {V12_CACHE}")

    # 評価
    print(f"\n{'='*70}")
    print(f"[評価結果]  フォーマット: 的中率%/ROI%")
    print(f"{'='*70}")
    evaluate("v11ev（ベースライン）",   race_records, "s_v11ev")
    evaluate("v12birth（binary+odds）", race_records, "s_v12birth")
    evaluate("v12birth_ev（EV回帰）",   race_records, "s_v12birth_ev")

    print(f"\n{'='*70}")
    print("完了")
