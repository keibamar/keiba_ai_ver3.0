import sys
import os
from datetime import date

# pycache を生成しない
sys.dont_write_bytecode = True

# web/src を import パスに追加
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# libs を追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LIBS_PATH = os.path.join(PROJECT_ROOT, "libs")
if LIBS_PATH not in sys.path:
    sys.path.insert(0, LIBS_PATH)

from generators.make_race_card_html import update_daily_html

if __name__ == "__main__":
    # race_day = date(2025, 12, 7)
    race_day = date.today()
    update_daily_html(race_day)