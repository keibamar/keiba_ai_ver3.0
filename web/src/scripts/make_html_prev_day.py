
import sys
from datetime import date, timedelta

# pycache を生成しない
sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\web\src\generators")

import make_race_card_html

if __name__ == "__main__":
    # race_day = date(2025, 12, 7)
    race_day = race_day = date.today() + timedelta(days=1)
    make_race_card_html.make_html_prev_day(race_day)