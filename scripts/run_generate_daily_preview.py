"""前日展望記事の自動生成スクリプト（毎週金・土 21:00 実行想定）

処理内容:
1. 翌日が開催日かを確認
2. race_time_id_list から注目レースを自動選定
3. 前年同名レースの成績を自動取得
4. Claude API で前日展望記事を生成
5. public_html/diary/{pub_date}.html として保存（インデックスも更新）

bat/TodayRace/generate_daily_preview.bat から呼ばれる。
"""

import os
import sys
import warnings
from datetime import date, timedelta

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# .env 読み込み
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

import pandas as pd

from src.managers import race_card_dataset_manager as rcm
from src.logic.article_generator import daily_preview_article as dpa
from src.logic.html_generator import weekly_review_generator as wrg

# 会場コード → 場名
_PLACE = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def _select_featured_races(races: list[dict]) -> tuple[list[str], str, list[str]]:
    """注目レースを自動選定する

    Returns:
        (featured_names, main_race_str, venues)
    """
    _generic = ("未勝利", "新馬", "1勝クラス", "2勝クラス", "3勝クラス", "障害OP")
    grade_order = {"G1": 0, "G2": 1, "G3": 2}

    # スコアリング: グレード → 高い方が優先。同グレードなら後半レース（大きい race_num）優先
    def score(r):
        grade = r.get("grade", "")
        is_named = r["race_name"] and not any(g in r["race_name"] for g in _generic)
        if not is_named:
            return (99, 0)
        g_score = grade_order.get(grade, 3)
        return (g_score, -r["race_num"])

    notable = [r for r in races if r["race_name"] and not any(g in r["race_name"] for g in _generic)]
    if not notable:
        return [], "", []

    notable.sort(key=score)
    featured = notable[:2]

    names = [r["race_name"] for r in featured]
    venues = list(dict.fromkeys(r["venue"] for r in featured))  # 重複除去・順序保持
    main_race = "・".join(names)
    return names, main_race, venues


def run(preview_date: date) -> bool:
    """指定日の前日展望を生成する

    Args:
        preview_date: 展望する日（翌日）

    Returns:
        True if generated, False if skipped
    """
    pub_date = preview_date - timedelta(days=1)
    print(f"[前日展望] pub_date={pub_date}  preview_date={preview_date}")

    # race_time_id_list 確認
    df_time = rcm.get_race_time_id_list_df(preview_date)
    if df_time.empty:
        print(f"  race_time_id_list なし: {preview_date} → スキップ")
        return False

    # 全レース情報を取得（race_dict 形式）
    races = []
    for _, row in df_time.iterrows():
        race_id = str(row["race_id"])
        race_name = str(row.get("race_name", ""))
        grade_csv = row.get("grade")
        grade_str = str(grade_csv) if pd.notna(grade_csv) else ""
        place_code = race_id[4:6]
        venue = _PLACE.get(place_code, place_code)
        race_num = int(race_id[10:12])
        info = dpa._get_race_info(race_id)
        if not grade_str and info.get("grade"):
            grade_str = info["grade"]
        races.append({
            "race_id": race_id,
            "race_name": race_name,
            "race_num": race_num,
            "venue": venue,
            "grade": grade_str,
            "info": info,
        })

    # 注目レースを自動選定
    featured_names, main_race, venues = _select_featured_races(races)
    if not featured_names:
        print(f"  注目レースなし → スキップ")
        return False

    print(f"  注目レース: {main_race}")
    print(f"  開催場: {venues}")

    # 既に同日の前日展望が存在する場合は再生成（上書き）
    out_html = os.path.join(wrg.DIARY_DIR, pub_date.strftime("%Y%m%d") + ".html")
    if os.path.exists(out_html):
        print(f"  既存ファイルを上書き: {out_html}")

    # データ収集
    data = dpa.collect_preview_data(preview_date, featured_race_names=featured_names)

    # 記事生成
    print("  Claude API で記事を生成中...")
    from src.logic.article_generator.claude_api_client import generate_article_text
    from src.logic.article_generator.daily_preview_article import _SYSTEM_PROMPT
    prompt = dpa.build_prompt(data)
    article_text = generate_article_text(_SYSTEM_PROMPT, prompt)
    print(f"  生成文字数: {len(article_text)}")

    # HTML 保存
    wrg.make_weekly_review_page(
        sat_date=preview_date,
        sun_date=None,
        article_text=article_text,
        data=data,
        article_type="前日展望",
        main_race=main_race,
        pub_date=pub_date,
        venues=venues,
    )
    print(f"  完了: {out_html}")
    return True


if __name__ == "__main__":
    preview_date = date.today() + timedelta(days=1)
    run(preview_date)
    print("Done:", preview_date)
