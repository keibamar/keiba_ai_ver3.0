"""race_schedule（race_calendar）データセットの構造定義

Dataset層はDataFrameの構造定義・変換のみを担当し、CSVの読み書き(I/O)は
src/managers/race_schedule_dataset_manager.py が担う。
"""

# race_calendar CSVの列
CALENDAR_COLUMNS = ["month", "day", "course", "times", "days"]

# race_id = 開催年(4桁) + 開催場id(2桁) + 開催回(2桁) + 開催日目(2桁) + レース番号(2桁)
RACE_ID_COURSE_DIGITS = 2
RACE_ID_TIMES_DIGITS = 2
RACE_ID_DAYS_DIGITS = 2
RACE_ID_RACE_NO_DIGITS = 2

# 1日あたりのレース数
RACES_PER_DAY = 12

# 開催回(times)・開催日目(days)の最大値（年間の全組み合わせ生成に使用）
MAX_TIMES_PER_YEAR = 6
MAX_DAYS_PER_TIMES = 12
