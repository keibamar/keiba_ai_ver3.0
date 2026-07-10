"""高配当的中ハイライト（画像付きX投稿）の検出・投稿を担うモジュール

ai_performance_calculator._race_detail_summaryが返す単勝/複勝の的中・配当を使い、
配当がrate_gauge_html.BET_RESULT_BIG_PAYOUT_THRESHOLD以上だったレースを検出し、
highlight_image_generatorで画像を生成してXに投稿する（土日18:00の予想結果投稿の
直後に実行する想定。run_today_race_returns.py参照）。
"""

import re
from datetime import date

from src.config.constants import NAME_LIST, PLACE_LIST
from src.config.lists import SYMBOL_LIST
from src.logic.calculators import ai_performance_calculator as calc
from src.logic.html_generator.rate_gauge_html import BET_RESULT_BIG_PAYOUT_THRESHOLD
from src.managers import race_card_dataset_manager, race_info_dataset_manager, race_schedule_dataset_manager
from src.output import highlight_image_generator, prediction_publisher

TRIO_BIG_PAYOUT_THRESHOLD = 10000.0

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


def _highlight_post_text(race_day, hit, place_name):
    """ハイライト投稿の本文を返す"""
    return (
        "🎯的中！\n"
        f"{place_name}{hit['race_num']}R {hit['race_name']}\n"
        f"{hit['bet_type_label']} {hit['payout']:.0f}円\n"
        f"AI本命: {hit['pick_name']}\n"
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
        text = _highlight_post_text(race_day, hit, place_name)
        prediction_publisher.post_text_with_image(text, image_path)
        print(f"高配当ハイライト投稿: {place_name}{hit['race_num']}R {hit['bet_type_label']} {hit['payout']:.0f}円")


def _trio_hit_for_summary(race_id, race_day, pred_df, box_num=5):
    """三連複5頭BOX的中の (payout, symbols_str) を返す、なければNone

    的中馬の予想記号を予想順（◎○▲△☆）に並べて返す。
    """
    returns_df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
    if returns_df.empty:
        return None

    sorted_df = pred_df.sort_values("score", ascending=False).reset_index(drop=True)
    top5_nums_ordered = sorted_df["馬番"].astype(int).tolist()[:box_num]
    top5_set = set(top5_nums_ordered)
    rank_of = {n: i for i, n in enumerate(top5_nums_ordered)}

    trio_df = returns_df[returns_df["式別"] == "三連複"].reset_index(drop=True)
    for i in range(len(trio_df)):
        num_list = [int(n) for n in re.findall(r"\d+", str(trio_df.at[i, "馬番"]))]
        if all(n in top5_set for n in num_list):
            payout_raw = trio_df.at[i, "配当"]
            if isinstance(payout_raw, str):
                payout_raw = re.sub(r"\D", "", payout_raw)
            payout = float(payout_raw)
            ranked = sorted(num_list, key=lambda n: rank_of.get(n, 99))
            symbols = "".join(SYMBOL_LIST[rank_of[n]] for n in ranked if n in rank_of)
            return payout, symbols
    return None


def collect_high_payout_hits(race_day):
    """当日の高配当的中（単勝/複勝≥1000円、三連複5頭BOX≥10000円）を全場分まとめて返す

    Args:
        race_day (date): レース開催日
    Returns:
        list[dict]: [{"place_id", "race_num", "race_name", "pick_name", "pick_num",
            "pick_pop", "win_payout"(or None), "place_payout"(or None),
            "trio_payout"(or None), "trio_symbols"(or None)}, ...]
    """
    hits = []
    for place_id in range(1, len(PLACE_LIST) + 1):
        race_id_list = sorted(race_schedule_dataset_manager.get_daily_id(place_id, race_day))
        if not race_id_list:
            continue
        time_id_df = race_card_dataset_manager.get_race_time_id_list_df(race_day)
        race_names = (
            dict(zip(time_id_df["race_id"].astype(str), time_id_df["race_name"]))
            if not time_id_df.empty else {}
        )
        for race_id in race_id_list:
            try:
                pred_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
                if pred_df.empty or "score" not in pred_df.columns:
                    continue
                detail = calc._race_detail_summary(race_day, race_id)
                if detail is None or detail.get("pick_scratched"):
                    continue

                pick_row = pred_df.sort_values("score", ascending=False).iloc[0]
                pick_num = int(pick_row["馬番"])
                race_num = calc.parse_race_id(race_id)["race_num"]
                race_name = race_names.get(str(race_id), "")

                win_big = detail["win_hit"] and detail["win_payout"] and detail["win_payout"] >= BET_RESULT_BIG_PAYOUT_THRESHOLD
                place_big = detail["place_hit"] and detail["place_payout"] and detail["place_payout"] >= BET_RESULT_BIG_PAYOUT_THRESHOLD
                trio_result = _trio_hit_for_summary(race_id, race_day, pred_df)
                trio_big = trio_result is not None and trio_result[0] >= TRIO_BIG_PAYOUT_THRESHOLD

                if not (win_big or place_big or trio_big):
                    continue
                hits.append({
                    "place_id": place_id,
                    "race_num": race_num,
                    "race_name": race_name,
                    "pick_name": detail["pick_name"],
                    "pick_num": pick_num,
                    "pick_pop": detail.get("pick_pop"),
                    "win_payout": detail["win_payout"] if win_big else None,
                    "place_payout": detail["place_payout"] if place_big else None,
                    "trio_payout": trio_result[0] if trio_big else None,
                    "trio_symbols": trio_result[1] if trio_big else None,
                })
            except Exception:
                continue
    hits.sort(key=lambda h: (h["place_id"], h["race_num"]))
    return hits


def post_daily_high_payout_summary(race_day=date.today()):
    """当日の高配当的中を1投稿にまとめてXに投稿する

    単勝/複勝は1000円以上、三連複5頭BOXは10000円以上が対象。
    対象がなければ投稿しない。

    Args:
        race_day (date): レース開催日（初期値: 今日）
    """
    hits = collect_high_payout_hits(race_day)
    if not hits:
        print("高配当的中なし（まとめ投稿なし）")
        return

    lines = ["🏆 本日の高配当的中", ""]
    for hit in hits:
        place_name = NAME_LIST[hit["place_id"] - 1]
        lines.append(f"{place_name}　{hit['race_num']}R　{hit['race_name']}")
        if hit["win_payout"] is not None or hit["place_payout"] is not None:
            pop_str = f"{hit['pick_pop']}人気　" if hit.get("pick_pop") else ""
            num_str = f"馬番{hit['pick_num']}　"
            payout_parts = []
            if hit["win_payout"] is not None:
                payout_parts.append(f"単勝{hit['win_payout']:.0f}円")
            if hit["place_payout"] is not None:
                payout_parts.append(f"複勝{hit['place_payout']:.0f}円")
            lines.append(f"◎{hit['pick_name']}　{pop_str}{num_str}{'/'.join(payout_parts)}")
        if hit["trio_payout"] is not None:
            lines.append(f"3連複{hit['trio_symbols']}　{hit['trio_payout']:.0f}円")
        lines.append("")

    lines.append("#MAR競馬予想 #競馬AI")
    prediction_publisher.post_text("\n".join(lines))
    print(f"高配当まとめ投稿完了（{len(hits)}件）")
