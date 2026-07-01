"""週次の定期X投稿（週末プレビュー・週末まとめ）を担うモジュール

高配当的中時の即時投稿（highlight_report）とは別に、投稿する話題が無い曜日でも
定期的にWebページへの導線を作るための2つの投稿を提供する。
  - post_weekend_preview: 金曜（投稿する話題が少ない曜日）に、今週末の注目レースを
    予告してHOMEへの来訪を促す。
  - post_weekend_summary: 火曜（週末の結果が出た直後）に、土日のAI成績
    （単勝/複勝の的中率・回収率）をまとめてAI成績ページへの来訪を促す。
"""

from datetime import date, timedelta

from src.config.constants import NAME_LIST
from src.logic.calculators import ai_performance_calculator as calc
from src.logic.html_generator.site_nav_html import SITE_URL
from src.managers import ai_performance_dataset_manager as m
from src.output import prediction_publisher


def post_weekend_preview(today=None):
    """今週末の注目レース（メインレース）をXに投稿し、HOMEへの来訪を促す

    出馬表（出走馬一覧）がまだ無い場合は投稿しない（calc.get_week_main_races_with_course
    が空リストを返すケース。投稿する内容が無いため）。

    Args:
        today (date | None): 基準日（初期値: 今日）。
    """
    today = today or date.today()
    races = calc.get_week_main_races_with_course(today)
    if not races:
        print("今週末の注目レースがまだ無いため、プレビュー投稿をスキップしました。")
        return

    lines = ["📅 今週末の注目レース", ""]
    for race in races:
        place_name = NAME_LIST[race["place_id"] - 1]
        weekday_kanji = "土" if race["race_day"].weekday() == 5 else "日"
        lines.append(f"・{weekday_kanji} {place_name}11R {race['race_name']}")
    lines.extend([
        "",
        "AI予想・出馬表はこちら👇",
        f"{SITE_URL}/",
        "#MAR競馬予想 #競馬AI",
    ])
    prediction_publisher.post_text("\n".join(lines))
    print("週末プレビュー投稿完了")


def post_weekend_summary(today=None):
    """直近の土日（週末）のAI成績（単勝/複勝の的中率・回収率）をXに投稿する

    結果反映には数日かかる想定のai_performance_calculator.current_results_weekend_end
    （水曜まで前の週末を指す）とは違い、この投稿は火曜実行を前提に、当日から見て
    直近の土日をそのまま対象にする（土日18:00の予想結果投稿の時点で確定済みのため）。
    対象レースが無い（n=0）場合は投稿しない。

    Args:
        today (date | None): 基準日（初期値: 今日）。
    """
    today = today or date.today()
    sunday = today - timedelta(days=(today.weekday() + 1) % 7)
    saturday = sunday - timedelta(days=1)

    df = m.get_ai_performance_dataset()
    weekend_df = m.filter_by_date_range(df, saturday, sunday)
    performance = m.aggregate(weekend_df)
    win, place = performance["win"], performance["place"]
    if win["n"] == 0:
        print("直近の週末のAI成績データが無いため、まとめ投稿をスキップしました。")
        return

    text = (
        f"📊 {saturday.strftime('%m/%d')}〜{sunday.strftime('%m/%d')}のMARの予想結果\n\n"
        f"単勝 的中率{win['hit_rate']:.1f}% 回収率{win['return_rate']:.1f}%\n"
        f"複勝 的中率{place['hit_rate']:.1f}% 回収率{place['return_rate']:.1f}%\n"
        f"(対象{win['n']}件)\n\n"
        "詳しいデータはこちら👇\n"
        f"{SITE_URL}/\n"
        "#MAR競馬予想 #競馬AI"
    )
    prediction_publisher.post_text(text)
    print("週末まとめ投稿完了")
