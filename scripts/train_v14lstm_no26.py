"""v14lstm_no26 モデル学習スクリプト（Sequence LSTM）

v8データセット（98列）を使用し PyTorch の LSTM + MLP でモデルを学習する。

入力構造:
  - 過去5走の時系列（5ステップ × 8特徴）: 古い走→新しい走の順で LSTM 入力
      各ステップ: [time_df_course, time_df_class, ninki, result, agari, margin, corner_ratio, corner_chase]
  - 血統3列（father_cat / mother_father_cat / paternal_gf_cat）: Embedding(各 embed_dim 次元)
  - 現在レース特徴54列（race_id・血統3列・時系列40列を除いた残り）: そのまま入力

学習:
  - 損失: BCEWithLogitsLoss + オッズ重み付き
  - 最適化: AdamW
  - スケジューラ: CosineAnnealingLR
  - Optuna でハイパーパラメータ探索（15 trials）
  - GPU 使用（CUDA）

サフィックス: _v14lstm_no26
実行: python scripts/train_v14lstm_no26.py
"""

import os, sys, traceback, warnings, pickle
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")

import name_header
from src.config import paths as paths_v3
from src.PredictionModels.LightGBM.prediction import _band_lengths
from src.PredictionModels.LightGBM.make_dataset_v5 import build_pedigree_vocab
from src.PredictionModels.LightGBM.make_dataset_v8 import load_dataset_v8, index_v8
from src.PredictionModels.LSTM.sequence_lstm import SequenceLSTM, SEQ_LEN, SEQ_FEATURES

TRAIN_YEARS   = list(range(2020, 2026))
TARGET_VENUES = list(range(1, 11))
MODEL_SUFFIX  = "_v14lstm_no26"
N_TRIALS      = 15
EPOCHS        = 50
BATCH_SIZE    = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {DEVICE}")

EMBED_COLS = ["father_cat", "mother_father_cat", "paternal_gf_cat"]

# 過去走の時系列列名（5走前→1走前=oldest first, LSTM入力順序）
SEQ_COLS_FLAT = []
for _i in range(5, 0, -1):
    SEQ_COLS_FLAT += [
        f"time_df_course{_i}", f"time_df_class{_i}",
        f"ninki_{_i}", f"result_{_i}", f"agari_{_i}", f"margin_{_i}", f"corner_ratio_{_i}",
        f"corner_chase_{_i}",
    ]

# 現在レース特徴（race_id・embed・sequence を除いた残り = 54列）
CURRENT_COLS = [c for c in index_v8 if c != "race_id" and c not in EMBED_COLS and c not in SEQ_COLS_FLAT]
NUM_CURRENT  = len(CURRENT_COLS)
assert NUM_CURRENT == 54, f"NUM_CURRENT={NUM_CURRENT} (期待値 54)"
print(f"SEQ_COLS: {len(SEQ_COLS_FLAT)}, CURRENT_COLS: {NUM_CURRENT}")


def _to_tensors(df, flag_df, vocab_size):
    """DataFrame → 学習用テンソルに変換。"""
    # 時系列: (n, 5, 8)
    seq_vals = df[SEQ_COLS_FLAT].fillna(-1).values.astype(np.float32)
    seq = torch.tensor(seq_vals.reshape(-1, SEQ_LEN, SEQ_FEATURES), dtype=torch.float32)

    # 現在特徴: (n, 54)
    x_cur = torch.tensor(
        df[CURRENT_COLS].fillna(-1).values.astype(np.float32), dtype=torch.float32
    )

    # 血統 ID
    def _safe_id(col):
        arr = df[col].fillna(-1).values.astype(np.int64)
        arr = np.where(arr < 0, 0, arr)
        arr = np.where(arr > vocab_size, vocab_size, arr)
        return torch.tensor(arr, dtype=torch.long)

    f_ids  = _safe_id("father_cat")
    mf_ids = _safe_id("mother_father_cat")
    pgf_ids = _safe_id("paternal_gf_cat")

    y = torch.tensor(
        (flag_df["result_flag"].values == 4).astype(np.float32), dtype=torch.float32
    )
    odds = torch.tensor(
        df["current_odds"].fillna(1).clip(lower=0.1).values.astype(np.float32),
        dtype=torch.float32,
    )
    return seq, f_ids, mf_ids, pgf_ids, x_cur, y, odds


def _make_weight(y, odds):
    return torch.where(y == 1, odds, torch.ones_like(odds))


def split_timeseries(race_data, race_flag, train_ratio=0.8):
    n = len(race_data)
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


def _train_epoch(model, optimizer, tr_seq, tr_f, tr_mf, tr_pgf, tr_x, tr_y, tr_w):
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    idx = torch.randperm(len(tr_y), device=DEVICE)
    total_loss, n_batches = 0.0, 0
    for start in range(0, len(tr_y), BATCH_SIZE):
        b = idx[start:start + BATCH_SIZE]
        logits = model(tr_seq[b], tr_f[b], tr_mf[b], tr_pgf[b], tr_x[b])
        loss = (criterion(logits, tr_y[b]) * tr_w[b]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _eval_loss(model, va_seq, va_f, va_mf, va_pgf, va_x, va_y, va_w):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        logits = model(va_seq, va_f, va_mf, va_pgf, va_x)
        loss = (criterion(logits, va_y) * va_w).mean().item()
    return loss


def _fit(params, vocab_size,
         tr_seq, tr_f, tr_mf, tr_pgf, tr_x, tr_y, tr_w,
         va_seq, va_f, va_mf, va_pgf, va_x, va_y, va_w,
         epochs=EPOCHS, verbose=False):
    model = SequenceLSTM(
        vocab_size=vocab_size,
        seq_features=SEQ_FEATURES,
        num_current=NUM_CURRENT,
        embed_dim=params.get("embed_dim", 16),
        lstm_hidden=params.get("lstm_hidden", 64),
        lstm_layers=params.get("lstm_layers", 1),
        hidden=params.get("hidden", [128, 64]),
        dropout=params.get("dropout", 0.2),
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.get("lr", 1e-3), weight_decay=params.get("wd", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss, best_state, patience = float("inf"), None, 0
    for ep in range(epochs):
        tr_loss = _train_epoch(model, optimizer, tr_seq, tr_f, tr_mf, tr_pgf, tr_x, tr_y, tr_w)
        va_loss = _eval_loss(model, va_seq, va_f, va_mf, va_pgf, va_x, va_y, va_w)
        scheduler.step()
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
        if verbose:
            print(f"    ep{ep+1:3d}: tr={tr_loss:.4f} va={va_loss:.4f} (best={best_loss:.4f})")
    if best_state:
        model.load_state_dict(best_state)
    return model, best_loss


def _tune(vocab_size,
          tr_seq, tr_f, tr_mf, tr_pgf, tr_x, tr_y, tr_w,
          va_seq, va_f, va_mf, va_pgf, va_x, va_y, va_w,
          n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "embed_dim":   trial.suggest_categorical("embed_dim", [8, 16, 32]),
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [32, 64, 128]),
            "lstm_layers": trial.suggest_categorical("lstm_layers", [1, 2]),
            "hidden":      trial.suggest_categorical("hidden", [
                [64, 32], [128, 64], [128, 64, 32],
            ]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "lr":      trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "wd":      trial.suggest_float("wd", 1e-5, 1e-3, log=True),
        }
        _, val_loss = _fit(
            params, vocab_size,
            tr_seq, tr_f, tr_mf, tr_pgf, tr_x, tr_y, tr_w,
            va_seq, va_f, va_mf, va_pgf, va_x, va_y, va_w,
            epochs=30,
        )
        return val_loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _save_model(model, vocab_size, params, place_id, race_type, length):
    type_str = "turf" if race_type == "芝" else "dirt"
    model_dir = os.path.join(paths_v3.PREDICTION_MODEL_PATH, name_header.PLACE_LIST[place_id - 1])
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{type_str}{length}_lambdarank_model{MODEL_SUFFIX}.pt")
    torch.save({
        "state_dict":   model.state_dict(),
        "vocab_size":   vocab_size,
        "params":       params,
        "num_current":  NUM_CURRENT,
        "seq_features": SEQ_FEATURES,
    }, path)
    return path


def train_all_courses(vocab_size):
    for place_id in TARGET_VENUES:
        place_name = name_header.PLACE_LIST[place_id - 1]
        print(f"\n{'='*55}\n[学習] {place_name}\n{'='*55}")

        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            print(f"\n  {race_type}{length}m ...")
            try:
                race_data, race_flag = pd.DataFrame(), pd.DataFrame()
                for year in TRAIN_YEARS:
                    for band_length in _band_lengths(place_id, race_type, length):
                        df, flag = load_dataset_v8(place_id, year, race_type, band_length)
                        if not df.empty and not flag.empty:
                            race_data = pd.concat([race_data, df])
                            race_flag = pd.concat([race_flag, flag])

                if race_data.empty:
                    print("    データなし: スキップ"); continue

                race_data = race_data.reset_index(drop=True)
                race_flag = race_flag.reset_index(drop=True)

                # SEQ_COLS_FLAT が全て存在するか確認
                missing = [c for c in SEQ_COLS_FLAT if c not in race_data.columns]
                if missing:
                    print(f"    列不足({missing[:3]}...): スキップ"); continue

                data_tr, data_va, flag_tr, flag_va = split_timeseries(race_data, race_flag)
                if data_va.empty:
                    print("    validationなし: スキップ"); continue

                print(f"    train={len(data_tr)}行 / val={len(data_va)}行")

                tr_seq, tr_f, tr_mf, tr_pgf, tr_x, tr_y, tr_odds = _to_tensors(data_tr, flag_tr, vocab_size)
                va_seq, va_f, va_mf, va_pgf, va_x, va_y, va_odds = _to_tensors(data_va, flag_va, vocab_size)
                tr_w = _make_weight(tr_y, tr_odds)
                va_w = _make_weight(va_y, va_odds)

                # GPU 転送
                for t in [tr_seq, tr_f, tr_mf, tr_pgf, tr_x, tr_y, tr_w,
                          va_seq, va_f, va_mf, va_pgf, va_x, va_y, va_w]:
                    t.data = t.to(DEVICE)

                best_params = _tune(
                    vocab_size,
                    tr_seq, tr_f, tr_mf, tr_pgf, tr_x, tr_y, tr_w,
                    va_seq, va_f, va_mf, va_pgf, va_x, va_y, va_w,
                )
                print(f"    best_params: {best_params}")

                model, _ = _fit(
                    best_params, vocab_size,
                    tr_seq, tr_f, tr_mf, tr_pgf, tr_x, tr_y, tr_w,
                    va_seq, va_f, va_mf, va_pgf, va_x, va_y, va_w,
                    epochs=EPOCHS, verbose=True,
                )
                path = _save_model(model.cpu(), vocab_size, best_params, place_id, race_type, length)
                print(f"    保存: {path}")

            except Exception:
                print(f"    ERROR: {race_type}{length}m")
                traceback.print_exc()


if __name__ == "__main__":
    print("=" * 55)
    print(f"v14lstm_no26  学習年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print("モデル: LSTM(過去5走×8特徴) + Embedding(血統×3) + MLP")
    print("損失: BCEWithLogitsLoss + オッズ重み付き")
    print(f"デバイス: {DEVICE}")
    print("=" * 55)

    print("\n[Step 1] 血統 vocab をロード...")
    vocab = build_pedigree_vocab()
    vocab_size = len(vocab)
    print(f"  vocab_size: {vocab_size}")

    print("\n[Step 2] モデル学習...")
    train_all_courses(vocab_size)

    print("\n" + "=" * 55 + "\n完了\n" + "=" * 55)
