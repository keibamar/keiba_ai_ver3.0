"""メール送信機能（旧実装）

新実装への移行に伴い、本体は src/output/prediction_publisher.py に移植済み。
このモジュールは後方互換のための re-export。
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.output.prediction_publisher import (  # noqa: E402,F401
    make_mime_text,
    process_txt_file,
    read_txt_file,
    send_email,
    send_gmail,
    send_race_pred,
)

# 実行部分
if __name__ == "__main__":
    file_path = "sample.txt"
    send_email(file_path)
