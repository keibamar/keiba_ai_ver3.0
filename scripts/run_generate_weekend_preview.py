"""週末重賞展望記事の自動生成スクリプト（毎週木曜 20:00 実行想定）

処理内容:
1. 翌土日の重賞レースを race_time_id_list から取得
2. 昨年の勝ち馬・コース傾向・出走馬（推定人気）を収集
3. Claude API で重賞展望記事を生成
4. public_html/diary/{pub_date}.html として保存（インデックスも更新）

bat/TodayRace/generate_weekend_preview.bat から呼ばれる。
"""

import os
import sys
import warnings
from datetime import date, timedelta

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

from src.logic.article_generator import weekend_preview_article as wpa
from src.logic.html_generator import weekly_review_generator as wrg

_PLACE = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def run(thu_date: date) -> bool:
    """指定の木曜日から週末重賞展望を生成する

    Args:
        thu_date: 木曜日の日付（実行日）

    Returns:
        True if generated, False if skipped
    """
    sat_date = thu_date + timedelta(days=2)
    sun_date = thu_date + timedelta(days=3)
    pub_date = thu_date  # 公開日は木曜（実行日）
    print(f"[週末重賞展望] pub_date={pub_date}  対象: {sat_date}〜{sun_date}")

    # データ収集
    data = wpa.collect_weekend_preview_data(sat_date, sun_date)
    graded = data["graded_races"]
    if not graded:
        print("  重賞レースなし → スキップ")
        return False

    print(f"  重賞レース数: {len(graded)}")
    for r in graded:
        entry_cnt = len(r.get("entries", []))
        entry_info = f"{entry_cnt}頭エントリー済み" if entry_cnt else "エントリー未取得"
        print(f"    {r['race_name']} ({r['grade']}) {r['venue']} / {entry_info}")

    # メインレース名・開催場
    main_race = "・".join(r["race_name"] for r in graded[:2])
    seen: set[str] = set()
    venues: list[str] = []
    for r in graded:
        v = r["venue"]
        if v not in seen:
            seen.add(v)
            venues.append(v)

    # 記事生成
    print("  Claude API で記事を生成中...")
    article_text, _ = wpa.generate(sat_date, sun_date)
    if not article_text:
        print("  記事テキストが生成できませんでした。")
        return False
    print(f"  生成文字数: {len(article_text)}")

    # HTML 保存
    wrg.make_weekly_review_page(
        sat_date=sat_date,
        sun_date=sun_date,
        article_text=article_text,
        data=data,
        article_type="週末展望",
        main_race=main_race,
        pub_date=pub_date,
        venues=venues,
    )
    out_html = os.path.join(wrg.DIARY_DIR, pub_date.strftime("%Y%m%d") + ".html")
    print(f"  完了: {out_html}")
    return True


if __name__ == "__main__":
    thu = date.today()
    run(thu)
    print("Done:", thu)
