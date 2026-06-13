import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LIBS_PATH = os.path.join(PROJECT_ROOT, "libs")
LEGACY_DATASETS_PATH = os.path.join(PROJECT_ROOT, "src", "legacy_datasets")

# 新構造(src.*)のimportと、旧libs/モジュールの直接importを両方可能にする
# src/legacy_datasets内のモジュール同士の相互import（horse_peds -> past_performance等）を
# 解決するため、src/legacy_datasetsもsys.pathに追加する
for path in (PROJECT_ROOT, LIBS_PATH, LEGACY_DATASETS_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)
