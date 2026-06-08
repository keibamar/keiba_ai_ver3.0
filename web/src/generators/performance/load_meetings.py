import csv
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))  # web
CONFIGS_PATH = os.path.join(PROJECT_ROOT, "src/config") 
sys.path.append(CONFIGS_PATH)

from config.path import RACE_CALENDAR_FOLDER_PATH

# -----------------------------
# 開催情報を読み込む
# -----------------------------
def load_meetings(year):
    meetings = {}  

    csv_path = os.path.join(RACE_CALENDAR_FOLDER_PATH, f"{year}_race_calendar.csv")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            month = int(row["month"])
            day = int(row["day"])
            course = int(row["course"])
            times = int(row["times"])
            days = int(row["days"])

            key = (year, course, times)

            if key not in meetings:
                meetings[key] = []

            meetings[key].append({
                "date": datetime(year, month, day),
                "day": days
            })

    # 日付順にソート
    for key in meetings:
        meetings[key].sort(key=lambda x: x["date"])

    return meetings