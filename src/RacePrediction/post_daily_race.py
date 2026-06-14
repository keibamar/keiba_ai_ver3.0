import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.scheduler.race_day_scheduler import (  # noqa: E402,F401
    post_daily_race_pred,
    post_pred_return,
    post_race_pred,
)

if __name__ == '__main__':
    post_daily_race_pred()
