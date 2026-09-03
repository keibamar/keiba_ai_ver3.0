"""eval_v11ev_no26.py

v11ev_no26 (114列特徴量 + EV回帰) を2026年データで評価。
v7/v11featモデルと的中率・回収率を比較する。

処理フロー:
  1. 2026年の v11 データセットを生成（未作成のコースのみ）
  2. v11ev モデルでスコアリング
  3. 実際の払戻データと突合して的中率・ROIを計算
  4. v11eval キャッシュ（v7/v11feat スコア）と比較

実行:
    python scripts/eval_v11ev_no26.py
"""

import glob, gc, itertools, os, pickle, sys, time, traceback, warnings
warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")

import lightgbm as lgb
import numpy as np
import pandas as pd

import name_header
from src.config import paths
from src.config.constants import PLACE_LIST
from src.PredictionModels.LightGBM.make_dataset_v5 import build_pedigree_vocab
from src.PredictionModels.LightGBM.make_dataset_v11 import (
    make_dataset_for_train_v11, load_dataset_v11,
)

BET_UNIT  = 100
EVAL_YEAR = 2026
V11EV_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_v11ev_{EVAL_YEAR}.pkl")
V11_CACHE   = os.path.join(PROJECT_ROOT, "logs", f"race_records_v11eval_{EVAL_YEAR}.pkl")
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
# 2026年データセット生成（未作成コースのみ）
# ============================================================

def _missing_courses_2026(place_id):
    out_dir = os.path.join(_DATASET_DIR, name_header.PLACE_LIST[place_id - 1])
    missing = []
    for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
        flag_path = os.path.join(
            out_dir, f"{EVAL_YEAR}_{race_type}{length}_ai_dataset_flag_v11.csv"
        )
        if not os.path.isfile(flag_path):
            missing.append((race_type, length))
    return missing


def generate_2026_datasets(vocab):
    print(f"\n[Step 1] {EVAL_YEAR}年 v11 データセット生成（未作成コースのみ）")
    for place_id in range(1, 11):
        missing = _missing_courses_2026(place_id)
        if not missing:
            continue
        pname = name_header.PLACE_LIST[place_id - 1]
        print(f"  {pname}: {len(missing)} コース")
        try:
            make_dataset_for_train_v11(place_id, year=EVAL_YEAR, vocab=vocab, course_filter=missing)
        except Exception:
            print(f"    ERROR: {pname}")
            traceback.print_exc()
        gc.collect()


# ============================================================
# 払戻データ取得
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
# v11ev スコア収集
# ============================================================

def collect_v11ev_records(vocab):
    print(f"\n[Step 2] v11ev スコア収集 ({EVAL_YEAR}年)")
    race_records = []
    t0 = time.time()
    processed = skipped = 0

    for place_id in range(1, 11):
        pname = name_header.PLACE_LIST[place_id - 1]
        print(f"\n  [{pname}]")

        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            df, flag = load_dataset_v11(place_id, EVAL_YEAR, race_type, length)
            if df.empty or flag.empty:
                continue

            try:
                model = _get_model(place_id, race_type, length, "_v11ev_no26")
            except FileNotFoundError as e:
                print(f"    モデルなし: {race_type}{length}m ({e})")
                continue

            feat_cols = [c for c in df.columns if c != "race_id"]
            X = df[feat_cols].fillna(-1)

            scores = model.predict(X)
            df = df.copy()
            df["_score"] = scores
            flag = flag.reset_index(drop=True)
            df = df.reset_index(drop=True)
            df["_flag"] = flag["result_flag"].values

            for rid, grp in df.groupby("race_id"):
                race_id_str = str(int(rid))
                grp = grp.reset_index(drop=True)

                ret_df = _get_return_df(race_id_str)
                res_df = _get_result_df(race_id_str)
                if ret_df.empty or res_df.empty:
                    skipped += 1
                    continue
                if "着順" not in res_df.columns:
                    skipped += 1
                    continue

                nums = pd.to_numeric(res_df["着順"], errors="coerce")
                winner_set = set(res_df[nums == 1]["馬番"].astype(str))
                top3_set   = set(res_df[nums <= 3]["馬番"].astype(str))
                san_row    = ret_df[ret_df["式別"] == "三連複"]
                san_winner = (
                    set(san_row["馬番"].iloc[0].split("-")) if not san_row.empty else None
                )
                san_odds = int(san_row["配当"].iloc[0]) if not san_row.empty else None

                # 馬番の復元: result_flagで判断 or res_dfから取得
                # grp内の馬番順はdf_course順（res_dfの行順）
                # res_dfの馬番を順に対応させる
                if len(grp) != len(res_df):
                    skipped += 1
                    continue

                # res_dfの馬番を行順で取得
                umabans = [str(int(float(u))) for u in res_df["馬番"].tolist()]
                odds_map = dict(zip(
                    res_df["馬番"].astype(str),
                    pd.to_numeric(res_df["単勝"], errors="coerce")
                ))
                fuku_rets = {u: _get_return(ret_df, "複勝", u) for u in umabans}
                tan_ret   = _get_return(ret_df, "単勝")

                s_v11ev = grp["_score"].values

                order   = np.argsort(-s_v11ev)
                top5    = [umabans[i] for i in order[:5] if i < len(umabans)]
                san_combs = list(itertools.combinations(top5, 3))

                race_records.append({
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
                    "umabans":    umabans,
                    "s_v11ev":    s_v11ev,
                })
                processed += 1

        if processed % 100 == 0 and processed > 0:
            print(f"    {processed}R収集済み ({time.time()-t0:.0f}秒)")

    print(f"\n  収集完了: {processed}R / スキップ: {skipped}R ({time.time()-t0:.0f}秒)")
    return race_records


# ============================================================
# 評価
# ============================================================

def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


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


def evaluate(label, records, score_fn, width=58):
    st = Stats()
    for r in records:
        scores = score_fn(r)
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
    print("=" * 60)
    print(f"eval_v11ev_no26  評価年: {EVAL_YEAR}")
    print("特徴量: v11（114列）  目的関数: EV回帰")
    print("=" * 60)

    # v11eval キャッシュ（v7/v11feat ベースライン）
    v11_records_by_id = {}
    if os.path.exists(V11_CACHE):
        print(f"\nv11eval キャッシュ読み込み: {V11_CACHE}")
        with open(V11_CACHE, "rb") as f:
            v11_recs = pickle.load(f)
        # race_idがないケースがあるのでスキップ
        for r in v11_recs:
            if "race_id" in r:
                v11_records_by_id[r["race_id"]] = r
        print(f"  {len(v11_recs)}R 読み込み（race_id有: {len(v11_records_by_id)}R）")
    else:
        print(f"  v11eval キャッシュなし: ベースライン比較はスキップ")

    # v11ev スコア収集（キャッシュ優先）
    if os.path.exists(V11EV_CACHE):
        print(f"\nv11ev キャッシュ読み込み: {V11EV_CACHE}")
        with open(V11EV_CACHE, "rb") as f:
            race_records = pickle.load(f)
        print(f"  {len(race_records)}R")
    else:
        print("\n血統vocab読み込み中...")
        vocab = build_pedigree_vocab()
        print(f"  {len(vocab)}種類")

        generate_2026_datasets(vocab)

        race_records = collect_v11ev_records(vocab)

        with open(V11EV_CACHE, "wb") as f:
            pickle.dump(race_records, f)
        print(f"  キャッシュ保存: {V11EV_CACHE}")

    # ベースライン（v11eval cache）と突合
    merged = []
    for r in race_records:
        rid = r.get("race_id", "")
        base = v11_records_by_id.get(rid, {})
        merged.append({**r,
                        "s_v7":  base.get("s_v7"),
                        "s_v11": base.get("s_v11")})

    W = 58
    SEP = "=" * 126
    print(f"\n{SEP}")
    print(f"  {'モデル':<{W}}  {'件数':>6}  {'単勝的中/回収':>15}  {'複勝的中/回収':>15}  {'3連複的中/回収':>16}")
    print(SEP)

    # v11ev 単独
    evaluate("v11ev_no26（EV回帰・114列）",
             race_records,
             lambda r: r["s_v11ev"])

    # v7 ベースライン（キャッシュから）
    v7_recs = [r for r in merged if r.get("s_v7") is not None]
    if v7_recs:
        evaluate("v7odds_binary_no26（ベースライン）",
                 v7_recs,
                 lambda r: r["s_v7"])
        # v11feat ベースライン
        v11f_recs = [r for r in merged if r.get("s_v11") is not None]
        if v11f_recs:
            evaluate("v11feat_no26（従来・91列）",
                     v11f_recs,
                     lambda r: r["s_v11"])

    print(f"\n--- v11ev × v7 ブレンド αスイープ ---")
    blend_recs = [r for r in merged if r.get("s_v7") is not None]
    if blend_recs:
        for alpha in [round(a * 0.1, 1) for a in range(0, 11)]:
            evaluate(f"v11ev×v7 α={alpha:.1f}", blend_recs,
                     lambda r, a=alpha: a * _norm(r["s_v11ev"]) + (1 - a) * _norm(r["s_v7"]))

    print(f"\n--- v11ev × v11feat ブレンド αスイープ ---")
    blend_recs2 = [r for r in merged if r.get("s_v11") is not None]
    if blend_recs2:
        for alpha in [round(a * 0.1, 1) for a in range(0, 11)]:
            evaluate(f"v11ev×v11f α={alpha:.1f}", blend_recs2,
                     lambda r, a=alpha: a * _norm(r["s_v11ev"]) + (1 - a) * _norm(r["s_v11"]))

    print(SEP)
