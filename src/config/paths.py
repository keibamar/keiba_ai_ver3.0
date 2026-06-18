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

# 予想エンジン関連データ（LightGBMのモデル・データセット・性能評価）
PREDICTION_DATA_PATH = os.path.join(DATA_PATH, "prediction")
PREDICTION_MODEL_PATH = os.path.join(PREDICTION_DATA_PATH, "models")
PREDICTION_DATASET_PATH = os.path.join(PREDICTION_DATA_PATH, "datasets")
PREDICTION_PERFORMANCE_PATH = os.path.join(PREDICTION_DATA_PATH, "performance")

# race_card（出馬表+score/rank）日次出力。旧 RACE_CARDS_PATH の新しい置き場所。
RACE_CARD_DATA_PATH = os.path.join(DATA_PATH, "race_card")

# HTML公開ディレクトリ（Git管理対象）
PUBLIC_HTML_PATH = os.path.join(PROJECT_ROOT, "public_html")
PUBLIC_HTML_RACES_PATH = os.path.join(PUBLIC_HTML_PATH, "races")
PUBLIC_HTML_ASSETS_PATH = os.path.join(PUBLIC_HTML_PATH, "assets")
PUBLIC_HTML_COURSES_PATH = os.path.join(PUBLIC_HTML_PATH, "courses")
PUBLIC_HTML_PERFORMANCE_PATH = os.path.join(PUBLIC_HTML_PATH, "performance")

# テキストデータ（レースカレンダー等の補助データ）
TEXTS_PATH = os.path.join(PROJECT_ROOT, "texts")
RACE_TIME_ID_LIST_PATH = os.path.join(TEXTS_PATH, "race_calendar", "race_time_id_list")

# 予想テキスト（メール本文・Xポスト用）。旧 TEXT_PATH + "race_prediction/" の新しい置き場所。
RACE_PREDICTION_TEXT_PATH = os.path.join(TEXTS_PATH, "race_prediction")  # {date}/{race_id}.txt

# 配当結果レポート（回収率テキスト）。旧 TEXT_PATH + "race_returns/" の新しい置き場所。
RACE_RETURN_REPORT_TEXT_PATH = os.path.join(TEXTS_PATH, "race_returns")  # {date}/{place}_pred_score.txt
