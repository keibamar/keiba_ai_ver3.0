"""週末レース振り返り記事（②）を生成するバッチ（bat/MakeHTML/make_weekly_review.bat から呼ばれる）

毎週火曜 20:00 実行想定。実行日（火曜）から見て直近の土日のデータを対象とする。
月曜開催がある週は、前週月曜も含める。

記事テキストは Claude API（claude-api-client）で生成し、
public_html/trend/review_YYYYMMDD.html として出力する。

依存: ANTHROPIC_API_KEY 環境変数（.env ファイルに設定）
"""

import os
import sys
import warnings
from datetime import date, timedelta

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# .env ファイルから環境変数を読み込む
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv が無くても env 変数が直接設定されていれば動く

from src.logic.article_generator import weekly_review_article  # noqa: E402
from src.logic.html_generator import weekly_review_generator  # noqa: E402


def _get_weekend_dates(today: date) -> tuple[date, date | None, date | None]:
    """実行日（火曜想定）から直近土日を返す"""
    # 直近の日曜（today から見て前の日曜）
    days_since_sunday = (today.weekday() + 1) % 7
    sun = today - timedelta(days=days_since_sunday)
    sat = sun - timedelta(days=1)

    # 月曜開催判定: 先週月曜（=sat の翌々日ではなく、先週の月曜）
    # sat の2日後が月曜。月曜にレースデータがあれば含める。
    mon = sun + timedelta(days=1)
    from src.managers import race_card_dataset_manager
    mon_has_races = not race_card_dataset_manager.get_race_time_id_list_df(mon).empty
    return sat, sun, (mon if mon_has_races else None)


if __name__ == "__main__":
    today = date.today()
    sat, sun, mon = _get_weekend_dates(today)

    print(f"対象: {sat} 〜 {sun}" + (f" 〜 {mon}" if mon else ""))

    article_text, data = weekly_review_article.generate(sat, sun, mon)

    if not article_text:
        print("記事テキストが生成できませんでした。終了します。")
        sys.exit(1)

    weekly_review_generator.make_weekly_review_page(sat, sun, article_text, data)
    print("週末振り返り記事の生成が完了しました。")
