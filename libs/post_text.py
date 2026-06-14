"""X(Twitter)投稿機能（旧実装）

新実装への移行に伴い、本体は src/output/prediction_publisher.py に移植済み。
このモジュールは後方互換のための re-export。
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.output.prediction_publisher import post_text_data, post_text_error  # noqa: E402,F401
