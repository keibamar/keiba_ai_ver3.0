import os
import sys

from datetime import date, timedelta
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import get_race_id

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.output.prediction_publisher import extract_top5_pred, make_race_text  # noqa: E402,F401
from src.output.return_report import (  # noqa: E402,F401
    make_return_text,
    write_place_hit_text,
    write_trio_hit_text,
    write_win_hit_text,
)

if __name__ == "__main__":
    place_id = 5
    race_day = date.today() - timedelta(2)
    race_id_list = get_race_id.get_daily_id(place_id, race_day)
    for race_id in race_id_list:
        make_race_text(race_day, race_id)