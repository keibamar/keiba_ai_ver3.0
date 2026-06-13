import os

# プロジェクトルート（このファイルから3階層上 = keiba_ai_ver3.0/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# データルート（ドメイン別フォルダで構成。.gitignore対象）
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

# ドメイン別データフォルダ（specifications/新設計.md のdata/構成）
RACE_SCHEDULE_DATA_PATH = os.path.join(DATA_PATH, "race_schedule")
RACE_RESULT_DATA_PATH = os.path.join(DATA_PATH, "race_result")
HORSE_DATA_PATH = os.path.join(DATA_PATH, "horse")
RACE_INFO_DATA_PATH = os.path.join(DATA_PATH, "race_info")

# HTML公開ディレクトリ（Git管理対象）
PUBLIC_HTML_PATH = os.path.join(PROJECT_ROOT, "public_html")

# テキストデータ（レースカレンダー等の補助データ）
TEXTS_PATH = os.path.join(PROJECT_ROOT, "texts")
RACE_TIME_ID_LIST_PATH = os.path.join(TEXTS_PATH, "race_calendar", "race_time_id_list")
