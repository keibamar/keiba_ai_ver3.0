"""高配当的中ハイライト（画像付きX投稿）の検出・投稿を担うモジュール

ai_performance_calculator._race_detail_summaryが返す単勝/複勝の的中・配当を使い、
配当がrate_gauge_html.BET_RESULT_BIG_PAYOUT_THRESHOLD以上だったレースを検出し、
highlight_image_generatorで画像を生成してXに投稿する（土日18:00の予想結果投稿の
直後に実行する想定。run_today_race_returns.py参照）。
"""

from datetime import date

from src.config.constants import NAME_LIST, PLACE_LIST
from src.logic.calculators import ai_performance_calculator as calc
from src.logic.html_generator.rate_gauge_html import BET_RESULT_BIG_PAYOUT_THRESHOLD
from src.managers import race_card_dataset_manager, race_schedule_dataset_manager
from src.output import highlight_image_generator, prediction_publisher

BET_TYPE_LABELS = {"win": "単勝", "place": "複勝"}


def find_big_hits(race_day):
    """指定日の的中レースのうち、配当が高額だったもの（単勝/複勝それぞれ判定）を返す

    AI本命馬が除外・取消だったレース（払い戻しで的中/不的中の対象外）は含めない。
    1レースで単勝・複勝の両方が高配当的中の場合は、それぞれ別の要素として含める。

    Args:
        race_day (date): レース開催日
    Returns:
        list[dict]: [{"race_id", "place_id", "race_num", "race_name", "bet_type",
            "bet_type_label", "payout", "pick_name"}, ...]（開催場・レース番号の昇順）
    """
    hits = []
    for place_id in range(1, len(PLACE_LIST) + 1):
        race_id_list = sorted(race_schedule_dataset_manager.get_daily_id(place_id, race_day))
        if not race_id_list:
            continue
        time_id_df = race_card_dataset_manager.get_race_time_id_list_df(race_day)
        race_names = (
            dict(zip(time_id_df["race_id"].astype(str), time_id_df["race_name"]))
            if not time_id_df.empty
            else {}
        )

        for race_id in race_id_list:
            try:
                detail = calc._race_detail_summary(race_day, race_id)
            except Exception:
                # data/race_result/側の既知のデータ品質問題（重複行）で例外になることがある。
                # そのレースだけ読み飛ばす（get_meeting_race_detailsと同じ方針）。
                continue
            if detail is None or detail["pick_scratched"]:
                continue

            race_num = calc.parse_race_id(race_id)["race_num"]
            for bet_type in ("win", "place"):
                payout = detail[f"{bet_type}_payout"]
                if detail[f"{bet_type}_hit"] and payout is not None and payout >= BET_RESULT_BIG_PAYOUT_THRESHOLD:
                    hits.append({
                        "race_id": race_id,
                        "place_id": place_id,
                        "race_num": race_num,
                        "race_name": race_names.get(str(race_id), ""),
                        "bet_type": bet_type,
                        "bet_type_label": BET_TYPE_LABELS[bet_type],
                        "payout": payout,
                        "pick_name": detail["pick_name"],
                    })

    hits.sort(key=lambda h: (h["place_id"], h["race_num"]))
    return hits


def _highlight_post_text(race_day, hit, place_name, place_key):
    """ハイライト投稿の本文を返す（該当レースの出馬表ページへ直リンクする）"""
    race_url = f"https://mar-keiba.com/races/{race_day.strftime('%Y%m%d')}/{place_key}R{hit['race_num']}.html"
    return (
        "🎯的中！\n"
        f"{place_name}{hit['race_num']}R {hit['race_name']}\n"
        f"{hit['bet_type_label']} {hit['payout']:.0f}円\n"
        f"AI本命: {hit['pick_name']}\n\n"
        f"{race_url}\n"
        "#MAR競馬予想 #競馬AI"
    )


def post_big_hit_highlights(race_day=date.today()):
    """指定日の高配当的中をすべて、画像付きでXに投稿する

    Args:
        race_day (date): レース開催日（初期値: 今日）
    """
    for hit in find_big_hits(race_day):
        place_name = NAME_LIST[hit["place_id"] - 1]
        place_key = PLACE_LIST[hit["place_id"] - 1]
        image_path = highlight_image_generator.highlight_image_path(race_day, hit["race_id"], hit["bet_type"])
        highlight_image_generator.make_hit_highlight_image(
            hit["bet_type_label"], hit["payout"], place_name, hit["race_num"], hit["race_name"],
            hit["pick_name"], image_path,
        )
        text = _highlight_post_text(race_day, hit, place_name, place_key)
        prediction_publisher.post_text_with_image(text, image_path)
        print(f"高配当ハイライト投稿: {place_name}{hit['race_num']}R {hit['bet_type_label']} {hit['payout']:.0f}円")
