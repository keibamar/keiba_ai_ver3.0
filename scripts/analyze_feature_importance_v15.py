"""analyze_feature_importance_v15.py

v15pace モデルの特徴量重要度を集計・分析する。

各競馬場・コースのモデルから重要度を集めて:
1. 特徴量グループ別の平均重要度
2. 上位20特徴量
3. v10で追加した特徴量カテゴリとの比較

実行: python scripts/analyze_feature_importance_v15.py
"""

import sys, os, warnings, glob
warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")

import numpy as np
import pandas as pd
import lightgbm as lgb
import name_header

from src.config import paths
from src.PredictionModels.LightGBM.make_dataset_v9 import index_v9

SUFFIX = "_v15pace_no26"
FEAT_COLS = [c for c in index_v9 if c != "race_id"]

# 特徴量グループ定義
FEAT_GROUPS = {
    "血統・過去3走ベース (v3)": [c for c in FEAT_COLS[:60]],
    "騎手勝率": ["jockey_win_rate", "jockey_place_rate"],
    "枠・馬番": ["枠番", "馬番"],
    "オッズ・人気": ["current_odds", "current_popularity"],
    "血統カテゴリ": ["father_cat", "mother_father_cat", "paternal_gf_cat"],
    "過去4・5走拡張 (v6)": [c for c in FEAT_COLS if c.startswith("time_df_course") or
                              c.startswith("time_df_class") or c.startswith("ninki_") or
                              c.startswith("result_") or c.startswith("agari_4") or
                              c.startswith("agari_5") or c.startswith("margin_") or
                              c.startswith("corner_ratio_4") or c.startswith("corner_ratio_5") or
                              c in ("rank_trend_5", "win_rate_recent5")],
    "斤量・馬体・頭数 (v7)": ["kinryo", "days_since_last_race", "n_horses_today",
                               "n_horses_1", "horse_weight_abs_1"],
    "コーナー追走・トレンド (v8)": [c for c in FEAT_COLS if c.startswith("corner_chase_") or
                                      c in ("agari_trend_5", "time_diff_trend_5")],
    "ペース適性 (v9新規)": [c for c in FEAT_COLS if c.startswith("agari_df_course_") or
                              c in ("corner_ratio_std5", "agari_std5")],
}


def load_models():
    models = {}
    for place_id in range(1, 11):
        place_name = name_header.PLACE_LIST[place_id - 1]
        model_dir = os.path.join(paths.PREDICTION_MODEL_PATH, place_name)
        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            type_str = "turf" if race_type == "芝" else "dirt"
            mp = os.path.join(model_dir, f"{type_str}{length}_lambdarank_model{SUFFIX}.txt")
            if os.path.isfile(mp):
                try:
                    models[(place_name, race_type, length)] = lgb.Booster(model_file=mp)
                except Exception as e:
                    print(f"  ロード失敗: {mp}: {e}")
    return models


print(f"v15pace_no26 モデルをロード中...")
models = load_models()
print(f"  {len(models)}件のモデルをロード")

# 全モデルの特徴量重要度を収集
all_importance = {}
for (place, rt, cl), model in models.items():
    fi = model.feature_importance(importance_type="gain")
    fn = model.feature_name()
    for name, imp in zip(fn, fi):
        all_importance.setdefault(name, []).append(imp)

# 平均重要度を計算
avg_imp = {k: np.mean(v) for k, v in all_importance.items()}
total_imp = sum(avg_imp.values())
rel_imp = {k: 100 * v / total_imp for k, v in avg_imp.items()}

# 上位20特徴量
print(f"\n{'='*70}")
print("【上位20特徴量（平均Gain重要度 %）】")
print(f"{'='*70}")
sorted_feats = sorted(rel_imp.items(), key=lambda x: -x[1])
for i, (name, imp) in enumerate(sorted_feats[:20], 1):
    bar = "█" * int(imp / 0.5) if imp > 0 else ""
    print(f"  {i:>2}. {name:<35} {imp:>6.2f}%  {bar}")

# グループ別集計
print(f"\n{'='*70}")
print("【特徴量グループ別 重要度シェア】")
print(f"{'='*70}")
group_imps = {}
assigned = set()
for group, cols in FEAT_GROUPS.items():
    valid_cols = [c for c in cols if c in rel_imp]
    group_imp = sum(rel_imp.get(c, 0) for c in valid_cols)
    group_imps[group] = group_imp
    assigned.update(valid_cols)

# 未分類
unassigned_imp = sum(imp for k, imp in rel_imp.items() if k not in assigned)
group_imps["その他"] = unassigned_imp

for group, imp in sorted(group_imps.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp / 2) if imp > 0 else ""
    print(f"  {group:<30} {imp:>6.1f}%  {bar}")

# v9 追加特徴量の重要度（ペース適性）
print(f"\n{'='*70}")
print("【v9 追加特徴量（ペース適性）の重要度詳細】")
print(f"{'='*70}")
v9_feats = ["agari_df_course_1", "agari_df_course_2", "agari_df_course_3",
            "agari_df_course_4", "agari_df_course_5", "corner_ratio_std5", "agari_std5"]
for f in v9_feats:
    imp = rel_imp.get(f, 0)
    rank = next((i+1 for i, (n, _) in enumerate(sorted_feats) if n == f), 999)
    print(f"  {f:<30} {imp:>6.2f}%  (rank {rank})")

# v10 追加予定特徴量と類似する既存特徴量
print(f"\n{'='*70}")
print("【v10 追加予定特徴量の関連既存特徴量（参考）】")
print(f"{'='*70}")
related = {
    "kinryo_diff_1 (斤量変化)": "kinryo",
    "hw_change_1 (馬体重変化)": "horse_weight_abs_1",
    "dist_change_1 (距離変化)": None,
    "same_course_cnt5 (同コース出走数)": "win_rate_recent5",
    "same_course_pr5 (同コース複勝率)": "win_rate_recent5",
}
for v10_feat, v9_ref in related.items():
    ref_imp = rel_imp.get(v9_ref, 0) if v9_ref else 0
    ref_rank = next((i+1 for i, (n, _) in enumerate(sorted_feats) if n == v9_ref), 999) if v9_ref else "-"
    if v9_ref:
        print(f"  {v10_feat:<35}  関連既存: {v9_ref} = {ref_imp:.2f}% (rank {ref_rank})")
    else:
        print(f"  {v10_feat:<35}  関連既存: なし（新規シグナル）")

print(f"\n総モデル数: {len(models)}")
