import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LIBS_PATH = os.path.join(PROJECT_ROOT, "libs")

# 新構造(src.*)のimportと、旧libs/モジュールの直接importを両方可能にする
for path in (PROJECT_ROOT, LIBS_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)
