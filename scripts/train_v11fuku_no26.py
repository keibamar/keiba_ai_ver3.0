"""train_v11fuku_no26.py — v11特徴量(114列) + 複勝EV専用回帰

目的関数: 複勝EV回帰（単勝成分なし）
  target = is_top3 × fuku_payout_odds
  - fuku_payout_odds = 複勝払戻(円) / 100 （FUKU_ODDS_CAP 上限）
  - 3着以外の馬はすべて target=0

評価指標: バリデーション期間の複勝ROI（Optuna / early-stopping）
TRAIN_YEARS: 2020〜2025（リークなし）
サフィックス: _v11fuku_no26

特徴:
  - 勝ち馬推奨ではなく「3着内推奨」に特化
  - 単勝EV成分がないため複勝価値のある穴馬も拾いやすい
  - v11ev（単勝特化）と組み合わせることで 単勝/複勝/3連複 の総合改善を狙う

実行:
    python scripts/train_v11fuku_no26.py
"""

import glob as _glob, os, sys, traceback, warnings
warnings.simplefilter("ignore")

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")

import name_header
from src.config.constants import USE_GPU_TRAINING, LIGHTGBM_DEVICE
from src.config import paths as paths_v3
from src.PredictionModels.LightGBM.prediction import _band_lengths
from src.PredictionModels.LightGBM.make_dataset_v11 import load_dataset_v11

TRAIN_YEARS   = list(range(2020, 2026))
TARGET_VENUES = list(range(1, 11))
MODEL_SUFFIX  = "_v11fuku_no26"
N_TRIALS      = 20
FUKU_ODDS_CAP = 30.0   # 複勝払戻上限（300円相当）

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}

_PLACE_DIR_NAMES = [
    "01_sapporo", "02_hakodate", "03_fukushima", "04_nigata",
    "05_tokyo",   "06_nakayama", "07_chukyo",    "08_kyoto",
    "09_hanshin", "10_kokura",
]


# ============================================================
# 複勝払戻ルックアップ構築（train_v11blend_no26 と同一ロジック）
# ============================================================

def _build_fuku_lookup(place_id, years):
    """
    Returns: {race_id_str: {result_flag(4/3/2): fuku_payout_odds}}
    result_flag: 4=1着, 3=2着, 2=3着
    fuku_payout_odds = 払戻(円) / 100 （FUKU_ODDS_CAP 上限）
    """
    lookup = {}
    place_dir = _PLACE_DIR_NAMES[place_id - 1]
    data_root  = paths_v3.DATA_PATH

    for year in years:
        annual_path = os.path.join(
            data_root, "race_result", place_dir, f"{year}_race_results.csv"
        )
        if not os.path.isfile(annual_path):
            continue
        try:
            res_df = pd.read_csv(annual_path, index_col=0)
        except Exception:
            continue

        ret_dir = os.path.join(
            data_root, "race_info", "race_returns", place_dir, str(year)
        )
        if not os.path.isdir(ret_dir):
            continue

        ret_frames = []
        for p in _glob.glob(os.path.join(ret_dir, "*.csv")):
            try:
                ret_frames.append(pd.read_csv(p, index_col=0))
            except Exception:
                pass

        if not ret_frames:
            continue

        all_ret  = pd.concat(ret_frames, ignore_index=False)
        fuku_all = all_ret[all_ret["式別"] == "複勝"].copy()
        fuku_all.index = fuku_all.index.map(
            lambda x: str(int(float(x))) if str(x).replace(".", "").isdigit() else str(x)
        )

        for rid, grp in res_df.groupby(res_df.index):
            rid_str   = str(int(float(str(rid))))
            fuku_race = fuku_all.loc[fuku_all.index == rid_str]
            if fuku_race.empty:
                continue

            try:
                nums = pd.to_numeric(grp["着順"], errors="coerce")
            except KeyError:
                continue

            race_lookup = {}
            for rank_num, flag in [(1, 4), (2, 3), (3, 2)]:
                horse_row = grp[nums == rank_num]
                if horse_row.empty:
                    continue
                try:
                    umaban = str(int(horse_row["馬番"].iloc[0]))
                except (KeyError, ValueError):
                    continue

                fuku_row = fuku_race[fuku_race["馬番"].astype(str) == umaban]
                if fuku_row.empty:
                    continue

                try:
                    payout = float(fuku_row["配当"].iloc[0])
                    race_lookup[flag] = min(payout / 100.0, FUKU_ODDS_CAP)
                except (ValueError, TypeError):
                    pass

            if race_lookup:
                lookup[rid_str] = race_lookup

    return lookup


# ============================================================
# ターゲット生成（複勝EV専用・単勝成分なし）
# ============================================================

def _make_fuku_ev_target(flag_df, race_ids, fuku_lookup):
    """複勝EV回帰ターゲット: top3馬のみ fuku_payout_odds、それ以外=0"""
    result_flags = flag_df["result_flag"].values
    fuku_ev      = np.zeros(len(flag_df))
    rid_strs     = [str(int(float(str(r)))) for r in race_ids]

    for i, (rid_str, rflag) in enumerate(zip(rid_strs, result_flags)):
        if rflag in (2, 3, 4) and rid_str in fuku_lookup:
            fuku_ev[i] = fuku_lookup[rid_str].get(int(rflag), 0.0)

    return fuku_ev


# ============================================================
# 評価指標: 複勝ROI
# ============================================================

def make_fuku_roi_eval(race_ids_val, flags_val, fuku_lookup):
    """バリデーション期間の複勝ROI（Optuna / early-stopping 共通）"""
    unique_ids = np.unique(race_ids_val)

    def feval(preds, dataset):
        total_fuku_pay = 0.0
        n_race = 0

        for rid in unique_ids:
            mask      = race_ids_val == rid
            pred_sub  = preds[mask]
            flags_sub = flags_val[mask]
            if len(pred_sub) == 0:
                continue

            top_idx     = np.argmax(pred_sub)
            picked_flag = int(flags_sub[top_idx])
            n_race += 1

            if picked_flag in (2, 3, 4):
                rid_str  = str(int(float(str(rid))))
                if rid_str in fuku_lookup:
                    fuku_pay = fuku_lookup[rid_str].get(picked_flag, 0.0)
                    total_fuku_pay += fuku_pay * 100.0

        roi = total_fuku_pay / (n_race * 100.0) if n_race > 0 else 0.0
        return "fuku_roi", roi, True

    return feval


# ============================================================
# モデル学習ユーティリティ
# ============================================================

def split_timeseries(race_data, race_flag, train_ratio=0.8):
    n     = len(race_data)
    split = int(n * train_ratio)
    while split < n - 1 and race_data.at[split - 1, "race_id"] == race_data.at[split, "race_id"]:
        split += 1
    if split >= n:
        return race_data, pd.DataFrame(), race_flag, pd.DataFrame()
    return (
        race_data.iloc[:split].reset_index(drop=True),
        race_data.iloc[split:].reset_index(drop=True),
        race_flag.iloc[:split].reset_index(drop=True),
        race_flag.iloc[split:].reset_index(drop=True),
    )


def _fit(params, ds_tr, ds_va, feval_fn, n_estimators):
    full_params = {
        "verbosity":      -1,
        "objective":      "regression",
        "num_iterations": n_estimators,
        **_DEVICE_KWARGS,
        **params,
    }
    return lgb.train(
        full_params,
        ds_tr,
        feval=feval_fn,
        valid_sets=[ds_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _tune(X_tr, y_tr, X_va, y_va, feval_fn, n_trials=20):
    min_cs_high = max(2, min(30, len(X_tr) // 10))

    def objective(trial):
        params = {
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 64),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, min_cs_high),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
        }
        n_est = trial.suggest_int("n_estimators", 100, 500)
        try:
            ds_tr_t = lgb.Dataset(X_tr, label=y_tr)
            ds_va_t = lgb.Dataset(X_va, label=y_va, reference=ds_tr_t)
            model   = _fit(params, ds_tr_t, ds_va_t, feval_fn, n_est)
            roi_val = model.best_score.get("valid_0", {}).get("fuku_roi", None)
            return roi_val if roi_val is not None else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _save_model(model, place_id, race_type, length):
    type_str  = "turf" if race_type == "芝" else "dirt"
    model_dir = os.path.join(paths_v3.PREDICTION_MODEL_PATH, name_header.PLACE_LIST[place_id - 1])
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{type_str}{length}_lambdarank_model{MODEL_SUFFIX}.txt")
    model.save_model(path)
    return path


# ============================================================
# メインループ
# ============================================================

def train_all_courses():
    for place_id in TARGET_VENUES:
        place_name = name_header.PLACE_LIST[place_id - 1]
        print(f"\n{'='*55}\n[学習] {place_name}\n{'='*55}")

        print(f"  複勝ルックアップ構築中 ...")
        fuku_lookup = _build_fuku_lookup(place_id, TRAIN_YEARS)
        print(f"  ルックアップ完了: {len(fuku_lookup):,} レース")

        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            print(f"\n  {race_type}{length}m ...")
            try:
                race_data, race_flag = pd.DataFrame(), pd.DataFrame()
                for year in TRAIN_YEARS:
                    for band_length in _band_lengths(place_id, race_type, length):
                        df, flag = load_dataset_v11(place_id, year, race_type, band_length)
                        if not df.empty and not flag.empty:
                            race_data = pd.concat([race_data, df])
                            race_flag = pd.concat([race_flag, flag])

                if race_data.empty:
                    print("    データなし: スキップ"); continue

                _num_cols = [c for c in race_data.columns if c != "race_id"]
                race_data[_num_cols] = race_data[_num_cols].fillna(-1)
                race_data = race_data.reset_index(drop=True)
                race_flag = race_flag.reset_index(drop=True)

                data_tr, data_va, flag_tr, flag_va = split_timeseries(race_data, race_flag)
                if data_va.empty:
                    print("    validationなし: スキップ"); continue

                rid_tr   = data_tr["race_id"].values
                rid_va   = data_va["race_id"].values
                flags_va = flag_va["result_flag"].values

                X_tr = data_tr.drop(columns=["race_id"])
                X_va = data_va.drop(columns=["race_id"])

                # 複勝EVのみ（単勝成分なし）
                y_tr = _make_fuku_ev_target(flag_tr, rid_tr, fuku_lookup)
                y_va = _make_fuku_ev_target(flag_va, rid_va, fuku_lookup)

                n_top3    = int((flag_tr["result_flag"] >= 2).sum())
                n_nonzero = int((y_tr > 0).sum())
                mean_pay  = float(y_tr[y_tr > 0].mean()) if n_nonzero > 0 else 0.0
                print(f"    train={len(X_tr):,}行 / val={len(X_va):,}行"
                      f" / top3={n_top3} / 複勝データ有={n_nonzero} / 平均複勝倍率={mean_pay:.2f}")

                if n_nonzero < 10:
                    print("    複勝データ不足: スキップ"); continue

                feval_fn = make_fuku_roi_eval(rid_va, flags_va, fuku_lookup)

                best_params = _tune(X_tr, y_tr, X_va, y_va, feval_fn, n_trials=N_TRIALS)
                print(f"    params: {best_params}")

                n_est  = best_params.pop("n_estimators", 300)
                ds_tr  = lgb.Dataset(X_tr, label=y_tr)
                ds_va  = lgb.Dataset(X_va, label=y_va, reference=ds_tr)
                model  = _fit(best_params, ds_tr, ds_va, feval_fn, n_est)
                path   = _save_model(model, place_id, race_type, length)
                print(f"    保存: {path}")

            except Exception:
                print(f"    ERROR: {race_type}{length}m")
                traceback.print_exc()


if __name__ == "__main__":
    print("=" * 55)
    print(f"v11fuku_no26  学習年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print("特徴量: v11（114列 = v10 + trainer/same_cond/ground_delta）")
    print("目的関数: 複勝EV専用回帰（単勝成分なし）")
    print(f"  target = is_top3 × fuku_payout_odds（上限{FUKU_ODDS_CAP}倍）")
    print("評価指標: 複勝ROI（Optuna / early-stopping）")
    print("=" * 55)

    train_all_courses()

    print("\n" + "=" * 55 + "\n完了\n" + "=" * 55)
