"""週末レース振り返り記事（②）のデータ収集・プロンプト生成・記事生成

毎週火曜実行。直近の土日（+ 月曜開催があれば月曜）のレース結果を集計し、
Claude API を呼び出してブログ記事テキストを生成する。

出力テキストはMarkdown形式。
HTML化は weekly_review_generator.make_weekly_review_page() が担う。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.logic.article_generator import claude_api_client
from src.managers import (
    ai_performance_dataset_manager as m,
    race_card_dataset_manager,
    race_result_dataset_manager,
)
from src.config.constants import NAME_LIST

_SYSTEM_PROMPT = """あなたは競馬情報サイト「MAR（馬予想AI）」のブログライターです。

【サイトの特徴】
- LightGBMを使ったAI予想モデル「MAR」が各レースの注目馬を選出
- 的中率重視（MAR-hit）・回収率重視（MAR-val）・バランス型（MAR）の3指数を提供
- 読者は競馬ファンで、毎週土日に馬券を楽しんでいる

【記事スタイル】
- 一人称は「私」または「MAR」
- 競馬ファン同士が感想を語り合うような親しみやすい文体
- データ・数字は具体的に（四捨五入して読みやすく）
- 当たった・外れたを正直に振り返る（良いことばかり書かない）
- 読者への問いかけを自然に盛り込む
- 文体例：「先週末もお疲れ様でした！今週の競馬はいかがだったでしょうか。」

【禁止事項】
- 馬券購入の断定的な推奨（「絶対に買うべき」等）
- 未確認情報の記述
- 同じ言い回しの連続使用"""

_ARTICLE_GUIDE = """
## 記事構成（合計3,000字程度）

### [リード]（200字）
今週末の競馬の雰囲気（天候・馬場など）から始め、読者に語りかけるように書く。

### 1. 今週末のMAR成績（500字）
- 単勝・複勝・3連複の的中率・回収率を具体的な数字で紹介
- 前週と比べてよかった点・残念だった点を正直に
- 「週次振り返り」ページ（傾向分析）とは重複させない
  → こちらは「馬券の成績・読者目線の結果」にフォーカス

### 2. 今週のハイライトレース（800字）
- 重賞レースの結果（あれば優先）
- 高配当が出たレース or MARが自信を持って推した馬の結果
- 印象的な勝ち馬のエピソードを交えて

### 3. MARスコア注目馬の振り返り（600字）
- MAR指数1位（rank_mar=1）だった馬の結果（2〜3頭）
- 当たった場合：なぜMAR指数が高かったのか振り返る
- 外れた場合：何が誤算だったか素直に分析する

### 4. 馬場・展開から読み解く今週の傾向（400字）
- 「週次振り返り」では取り上げないような、馬券視点での気づき
  （例：「荒れた馬場でも短距離は前が止まらなかった」など）

### 5. 来週への展望と読者への問いかけ（300字）
- 来週末の主な重賞をひとこと紹介
- 「今週の馬券結果はいかがでしたか？」等、読者が答えやすい問いかけで締める

---
上記構成を参考にしつつ、データに合わせて自然に調整してください。
Markdownの見出し（## や ###）を使って構成してください。
本文中に表（テーブル）は一切使わず、数字は文章に組み込んでください。
"""


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _get_winner(result_df: pd.DataFrame) -> dict | None:
    """レース結果DataFrameから1着馬の情報を返す"""
    if result_df.empty:
        return None
    first = result_df[result_df["着順"] == "1"]
    if first.empty:
        return None
    row = first.iloc[0]
    return {
        "name": row.get("馬名", ""),
        "odds": _safe_float(row.get("単勝", 0)),
        "popularity": row.get("人気", ""),
        "jockey": row.get("騎手", ""),
        "time": row.get("タイム", ""),
    }


def _get_race_mar_top(card_df: pd.DataFrame) -> dict | None:
    """出馬表DataFrameからMAR指数1位の馬情報を返す"""
    if card_df.empty or "rank_mar" not in card_df.columns:
        return None
    top = card_df[card_df["rank_mar"] == 1]
    if top.empty:
        # rank_mar が数値でないケース
        try:
            top = card_df.loc[card_df["rank_mar"].astype(float).idxmin():
                               card_df["rank_mar"].astype(float).idxmin()]
        except Exception:
            return None
    row = top.iloc[0]
    return {
        "name": row.get("馬名", ""),
        "idx_mar": _safe_float(row.get("idx_mar", 0)),
        "popularity": row.get("人気", ""),
        "odds": _safe_float(row.get("オッズ", 0)),
    }


def collect_weekend_data(sat_date: date, sun_date: date | None = None,
                         mon_date: date | None = None) -> dict:
    """土日（＋月曜）のレースデータをまとめて収集する

    Args:
        sat_date: 土曜日
        sun_date: 日曜日（None なら土曜のみ）
        mon_date: 月曜日（祝日開催がある週のみ）

    Returns:
        article生成に必要なデータを格納した dict
    """
    days = [d for d in [sat_date, sun_date, mon_date] if d is not None]

    race_summaries = []
    graded_races = []
    high_payout_race = None
    max_payout = 0.0

    for race_day in days:
        time_id_df = race_card_dataset_manager.get_race_time_id_list_df(race_day)
        if time_id_df.empty:
            continue

        for _, row in time_id_df.iterrows():
            race_id = str(row["race_id"])
            race_name = str(row.get("race_name", ""))
            grade = row.get("grade")

            card_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
            result_df = race_result_dataset_manager.get_race_id_result(race_id)

            winner = _get_winner(result_df)
            mar_top = _get_race_mar_top(card_df)

            mar_hit = (
                winner is not None
                and mar_top is not None
                and winner["name"] == mar_top["name"]
            )

            summary = {
                "race_id": race_id,
                "race_day": race_day.isoformat(),
                "race_name": race_name,
                "grade": grade if pd.notna(grade) else None,
                "winner": winner,
                "mar_top": mar_top,
                "mar_hit": mar_hit,
            }
            race_summaries.append(summary)

            if grade in ("G1", "G2", "G3"):
                graded_races.append(summary)

            if winner and winner["odds"] > max_payout:
                max_payout = winner["odds"]
                high_payout_race = summary

    # AI成績集計
    df_perf = m.get_ai_performance_dataset()
    end_date = mon_date or sun_date or sat_date
    weekend_df = m.filter_by_date_range(df_perf, sat_date, end_date)
    perf = m.aggregate(weekend_df)

    return {
        "sat_date": sat_date,
        "sun_date": sun_date,
        "mon_date": mon_date,
        "race_summaries": race_summaries,
        "graded_races": graded_races,
        "high_payout_race": high_payout_race,
        "max_payout": max_payout,
        "perf": perf,
    }


def _format_perf(perf: dict) -> str:
    """パフォーマンス集計結果を読みやすいテキストに変換する"""
    win = perf.get("win", {})
    place = perf.get("place", {})
    trio = perf.get("trio_box", {})
    n = win.get("n", 0)
    lines = [f"対象レース数: {n}レース"]
    if n > 0:
        lines.append(
            f"単勝: 的中率{win.get('hit_rate', 0):.1f}% / 回収率{win.get('return_rate', 0):.1f}%"
        )
        lines.append(
            f"複勝: 的中率{place.get('hit_rate', 0):.1f}% / 回収率{place.get('return_rate', 0):.1f}%"
        )
        lines.append(
            f"3連複（BOX）: 的中率{trio.get('hit_rate', 0):.1f}% / 回収率{trio.get('return_rate', 0):.1f}%"
        )
    return "\n".join(lines)


def _format_race_list(summaries: list[dict], limit: int = 8) -> str:
    """レース一覧を読みやすいテキストに変換する"""
    lines = []
    for s in summaries[:limit]:
        w = s.get("winner")
        mt = s.get("mar_top")
        grade_tag = f"【{s['grade']}】" if s.get("grade") else ""
        winner_str = (
            f"1着: {w['name']}（{w['popularity']}番人気 {w['odds']}倍）" if w else "1着: 結果未取得"
        )
        mar_str = ""
        if mt:
            hit_mark = "◎ MAR的中" if s["mar_hit"] else "✗ MAR外れ"
            mar_str = f" | MAR1位: {mt['name']}（指数{mt['idx_mar']:.0f}）→ {hit_mark}"
        lines.append(f"- {s['race_day']} {grade_tag}{s['race_name']}: {winner_str}{mar_str}")
    return "\n".join(lines) if lines else "（データなし）"


def build_prompt(data: dict) -> str:
    """記事生成用ユーザープロンプトを構築する"""
    sat = data["sat_date"]
    sun = data.get("sun_date")
    mon = data.get("mon_date")

    date_range = sat.strftime("%Y年%m月%d日（土）")
    if sun:
        date_range += f"〜{sun.strftime('%m月%d日（日）')}"
    if mon:
        date_range += f"〜{mon.strftime('%m月%d日（月）')}"

    graded_str = _format_race_list(data.get("graded_races", []))
    high_payout = data.get("high_payout_race")
    if high_payout and high_payout.get("winner"):
        high_payout_str = (
            f"{high_payout['race_name']} 1着: {high_payout['winner']['name']}"
            f"（{high_payout['winner']['odds']:.1f}倍 {high_payout['winner']['popularity']}番人気）"
        )
    else:
        high_payout_str = "（高配当レースなし）"

    all_races_str = _format_race_list(data.get("race_summaries", []), limit=20)
    perf_str = _format_perf(data.get("perf", {}))

    return f"""以下のデータをもとに、競馬ブログの週末振り返り記事を書いてください。

## 対象週末
{date_range}

## MARのAI成績
{perf_str}

## 重賞レース結果
{graded_str if graded_str else "（今週末に重賞なし）"}

## 最高配当レース
{high_payout_str}

## 全レース概要（MAR予想との照合）
{all_races_str}

---
{_ARTICLE_GUIDE}

※「週次傾向振り返り」（同じサイト内の別ページ）では馬場状態・展開傾向・上がり3F等のデータ面をすでに扱っています。
この記事はそちらと**重複しないよう**、馬券視点・読者体験・エピソード中心の内容にしてください。
"""


def generate(sat_date: date, sun_date: date | None = None,
             mon_date: date | None = None) -> tuple[str, dict]:
    """週末振り返り記事を生成する

    Args:
        sat_date: 土曜日
        sun_date: 日曜日
        mon_date: 月曜日（祝日開催がある場合）

    Returns:
        (article_markdown_text, data_context)
    """
    data = collect_weekend_data(sat_date, sun_date, mon_date)
    prompt = build_prompt(data)
    text = claude_api_client.generate_article_text(_SYSTEM_PROMPT, prompt)
    return text, data
