"""前日展望記事（③）のデータ収集・プロンプト生成・記事生成

金曜または土曜に実行。翌日のレースカードデータをもとに
Claude API を呼び出してプレビュー記事テキストを生成する。

出力テキストはMarkdown形式。
HTML化は weekly_review_generator.make_weekly_review_page() が担う。
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd

from src.logic.article_generator import claude_api_client
from src.managers import race_card_dataset_manager as rcm
from src.config import paths

_SYSTEM_PROMPT = """あなたは競馬情報サイト「MAR（まーる）」のコラムライターです。

【サイトの特徴】
- LightGBMを使ったAI予想モデル「MAR」が各レースの注目馬を選出
- 的中率重視（MAR-hit）・回収率重視（MAR-val）・バランス型（MAR）の3指数を提供
- 読者は競馬ファンで、毎週土日に競馬を楽しんでいる

【記事スタイル】
- 一人称は「私」または「MAR」
- 明日のレースへの期待感を共有する、前向きで親しみやすい文体
- 断定的な「この馬が勝つ」「買うべき」表現は避け「注目している」「面白そう」等を使う
- 読者への自然な問いかけを盛り込む

【指数・予想の表現について】
- 指数はあくまで「前日段階の数値」であることを自然に示す（断言しない）
- MAR-hitやMAR-valの最終予想は当日にならないとわからないことを、さりげなく示す
- 馬場状態・天候は前日予報ベースの評価であることを読者が分かるよう暗に示す
- 嘘をつかず、かつ過度に強調もしない

【フォーマット規則】
- レース名は「**オパールS**」のように太字にする
- 馬名は「**レッドキングリー**」のように太字にする
- Markdownの見出し（## や ###）を使う
- 本文中に表（テーブル）は一切使わない。数字は文章に自然に組み込む

【禁止事項】
- 馬券購入の断定的な推奨
- 未確認情報の記述
- 2歳戦の比率・頭数構成など、全体的なレース傾向のまとめ"""

_ARTICLE_GUIDE = """
## 記事構成（合計2,000〜2,500字程度）

### [リード]（150字）
明日の開催への期待感から書き出す。天候・馬場予報、開催場の雰囲気など。

### タイトル注目レース（1つめ）（650字）
データに「featured_race_1」として指定されたレースを取り上げる。
**必ず含める情報：**
- 開催場・何レース目・レース名・距離・コース条件
- このレースがどういうレースか（2〜3行の概要）
- 昨年の勝ち馬（渡されたデータにあれば）
- MAR注目馬の名前（太字）・父馬・母父・MAR指数（的中率軸/回収率軸）
- なぜ指数が高いか：血統・父のコース適性など渡されたデータから読み取れる理由を具体的に
- 馬場状態・天候が変わった場合の影響に軽く触れる

### タイトル注目レース（2つめ）（550字）
データに「featured_race_2」として指定されたレースを同様に取り上げる。

### その他の注目レース（350字）
残りの特別戦・オープン戦のうち1〜2鞍を簡潔に紹介する。
レース名・開催場・MAR注目馬名・指数程度にとどめる。

### 新馬戦ピックアップ（200字）
指数が突出した馬がいれば1頭だけ紹介。血統で理由を説明する。
突出した馬がいなければ省略する。

### 読者へのひとこと（150字）
「今日はどのレースに注目ですか？」等、読者が答えやすい問いかけで締める。

---
- 構成はデータに応じて自然に調整してよい
- Markdownの見出しを使う。本文中に表は使わない
- レース名・馬名は必ず太字
- 指数の高い理由は渡されたデータに基づいて説明し、憶測は「〜かもしれない」で示す
"""


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# 会場コード → 場名
_PLACE = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}
# 場名 → 結果CSVサブディレクトリ
_PLACE_DIR = {
    "札幌": "01_sapporo", "函館": "02_hakodate", "福島": "03_fukushima",
    "新潟": "04_nigata", "東京": "05_tokyo", "中山": "06_nakayama",
    "中京": "07_chukyo", "京都": "08_kyoto", "阪神": "09_hanshin",
    "小倉": "10_kokura",
}


def _get_race_info(race_id: str) -> dict:
    df = rcm.get_race_info_csv(race_id)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "race_type": str(row.get("race_type", "")),
        "course_len": str(row.get("course_len", "")),
        "weather": str(row.get("weather", "")),
        "ground_state": str(row.get("ground_state", "")),
        "class": str(row.get("class", "")),
        "grade": str(row.get("grade", "")) if pd.notna(row.get("grade")) else "",
    }


def _get_top_horses(card: pd.DataFrame, n: int = 3) -> list[dict]:
    if card.empty or "idx_mar" not in card.columns:
        return []
    top = card.nlargest(n, "idx_mar")
    horses = []
    for _, r in top.iterrows():
        idx_h = _safe_float(r.get("idx_hitrate", 0))
        idx_v = _safe_float(r.get("idx_value", 0))
        idx_m = _safe_float(r.get("idx_mar", 0))
        p_ai = _safe_float(r.get("p_ai", 0))
        p_mkt = _safe_float(r.get("p_market", 0))
        horses.append({
            "name": str(r.get("馬名", "")),
            "sex_age": str(r.get("性齢", "")),
            "jockey": str(r.get("騎手", "")),
            "trainer": str(r.get("厩舎", "")),
            "weight": str(r.get("馬体重(増減)", "")),
            "sire": str(r.get("peds_0", "")),
            "msire": str(r.get("peds_4", "")),
            "odds": str(r.get("オッズ", "---")),
            "popularity": str(r.get("人気", "?")),
            "idx_mar": idx_m,
            "idx_hitrate": idx_h,
            "idx_value": idx_v,
            "p_ai": p_ai,
            "p_market": p_mkt,
        })
    return horses


def _get_past_winner(race_name: str, venue: str, preview_date: date) -> dict | None:
    """前年同名レースの1着馬を取得する"""
    prev_year = preview_date.year - 1
    place_dir = _PLACE_DIR.get(venue)
    if not place_dir:
        return None

    result_csv = os.path.join(
        paths.DATA_PATH, "race_result", place_dir,
        f"{prev_year}_race_results.csv",
    )
    if not os.path.exists(result_csv):
        return None

    # time_id_list から race_id を探す（前年の10〜11月ごろ）
    place_code = next((k for k, v in _PLACE.items() if v == venue), None)
    if not place_code:
        return None

    target_race_id = None
    for day_offset in range(-30, 30):
        check_date = preview_date.replace(year=prev_year) + timedelta(days=day_offset)
        time_id_path = os.path.join(
            paths.RACE_TIME_ID_LIST_PATH,
            check_date.strftime("%Y%m%d") + ".csv",
        )
        if not os.path.exists(time_id_path):
            continue
        df_t = pd.read_csv(time_id_path, dtype=str)
        if "race_name" not in df_t.columns:
            continue
        match = df_t[
            (df_t["race_name"] == race_name) &
            (df_t["race_id"].str[4:6] == place_code)
        ]
        if not match.empty:
            target_race_id = match.iloc[0]["race_id"]
            break

    if not target_race_id:
        return None

    df_r = pd.read_csv(result_csv, dtype=str)
    winners = df_r[
        (df_r["Unnamed: 0"] == target_race_id) &
        (df_r["着順"] == "1")
    ]
    if winners.empty:
        return None

    w = winners.iloc[0]
    return {
        "year": prev_year,
        "name": str(w.get("馬名", "")),
        "sex_age": str(w.get("性齢", "")),
        "jockey": str(w.get("騎手", "")),
        "time": str(w.get("タイム", "")),
        "odds": str(w.get("単勝", "")),
        "popularity": str(w.get("人気", "")),
        "ground_state": str(w.get("ground_state", "")),
    }


def _format_horse_detail(h: dict) -> str:
    gap = h["p_ai"] - h["p_market"]
    gap_note = ""
    if gap > 0.05:
        gap_note = "（市場評価より高い・AI上振れ傾向）"
    elif gap < -0.05:
        gap_note = "（市場評価より低い・高人気）"

    diff = h["idx_value"] - h["idx_hitrate"]
    if diff > 10:
        idx_note = "→ 回収率重視で特に評価"
    elif diff < -10:
        idx_note = "→ 的中率重視で特に評価"
    else:
        idx_note = "→ 的中率・回収率ともバランス型"

    return (
        f"  - {h['name']}（{h['sex_age']}） 騎手:{h['jockey']} 厩舎:{h['trainer']}\n"
        f"    父:{h['sire']} 母父:{h['msire']} 馬体重:{h['weight']}\n"
        f"    MAR指数:{h['idx_mar']:.0f}（的中率軸:{h['idx_hitrate']:.0f} / 回収率軸:{h['idx_value']:.0f}）{idx_note}\n"
        f"    推定{h['popularity']}番人気 / オッズ:{h['odds']}{gap_note}"
    )


def collect_preview_data(
    preview_date: date,
    featured_race_names: list[str] | None = None,
) -> dict:
    """指定日のレースカードデータを収集する"""
    df_time = rcm.get_race_time_id_list_df(preview_date)
    if df_time.empty:
        return {
            "preview_date": preview_date,
            "races": [],
            "notable_races": [],
            "featured_races": [],
            "maiden_races": [],
        }

    races = []
    notable_races = []
    maiden_races = []

    for _, row in df_time.iterrows():
        race_id = str(row["race_id"])
        race_name = str(row.get("race_name", ""))
        race_time = str(row.get("race_time", ""))
        grade_csv = row.get("grade")
        grade_str = str(grade_csv) if pd.notna(grade_csv) else ""

        race_info = _get_race_info(race_id)
        if not grade_str and race_info.get("grade"):
            grade_str = race_info["grade"]

        card = rcm.get_race_cards(preview_date, race_id)
        top_horses = _get_top_horses(card, n=3) if not card.empty else []
        entry_count = len(card)

        place_code = race_id[4:6]
        venue = _PLACE.get(place_code, f"会場{place_code}")
        race_num = int(race_id[10:12])

        race_dict = {
            "race_id": race_id,
            "race_name": race_name,
            "race_num": race_num,
            "race_time": race_time,
            "venue": venue,
            "grade": grade_str,
            "entry_count": entry_count,
            "info": race_info,
            "top_horses": top_horses,
            "past_winner": None,
        }
        races.append(race_dict)

        _generic = ("未勝利", "新馬", "1勝クラス", "2勝クラス", "3勝クラス", "障害OP")
        is_named = race_name and not any(g in race_name for g in _generic)
        is_graded = grade_str in ("G1", "G2", "G3")

        if is_graded or is_named:
            notable_races.append(race_dict)

        if "新馬" in race_name:
            maiden_races.append(race_dict)

    # featured races（タイトルに含まれるレース）の過去成績を取得してマーク
    featured_races = []
    if featured_race_names:
        name_to_race = {r["race_name"]: r for r in notable_races}
        for fname in featured_race_names:
            r = name_to_race.get(fname)
            if r:
                past = _get_past_winner(fname, r["venue"], preview_date)
                r["past_winner"] = past
                featured_races.append(r)

    return {
        "preview_date": preview_date,
        "races": races,
        "notable_races": notable_races,
        "featured_races": featured_races,
        "maiden_races": maiden_races,
    }


def _format_past_winner(pw: dict | None) -> str:
    if not pw:
        return "（昨年データなし）"
    return (
        f"{pw['year']}年優勝: {pw['name']}（{pw['sex_age']}）"
        f" 騎手:{pw['jockey']} タイム:{pw['time']}"
        f" 単勝:{pw['odds']}倍 {pw['popularity']}番人気"
        f" 馬場:{pw['ground_state']}"
    )


def _format_race_block(r: dict, label: str = "") -> str:
    info = r.get("info", {})
    race_type = info.get("race_type", "")
    course_len = info.get("course_len", "")
    weather = info.get("weather", "")
    ground = info.get("ground_state", "")
    cls = info.get("class", "")
    grade = r.get("grade", "")

    grade_tag = f"【{grade}】" if grade else ""
    cond_str = f"{race_type}{course_len}m" if race_type and course_len else ""
    ground_str = f"天候:{weather} 馬場:{ground}" if weather or ground else ""
    cls_str = f"クラス:{cls}" if cls else ""
    past_str = _format_past_winner(r.get("past_winner"))

    header = f"### {label}{r['venue']} R{r['race_num']} 「{r['race_name']}」{grade_tag} {r['entry_count']}頭"
    lines = [
        header,
        f"発走: {r['race_time']} | {cond_str} | {ground_str} | {cls_str}",
        f"昨年成績: {past_str}",
        "MAR注目馬（上位3頭）:",
    ]
    for h in r.get("top_horses", []):
        lines.append(_format_horse_detail(h))
    return "\n".join(lines)


def _format_maiden_block(r: dict) -> str:
    info = r.get("info", {})
    race_type = info.get("race_type", "")
    course_len = info.get("course_len", "")
    top = r.get("top_horses", [])
    if not top:
        return ""
    h = top[0]
    cond_str = f"{race_type}{course_len}m" if race_type and course_len else ""
    return (
        f"- {r['venue']} R{r['race_num']} 新馬戦 {cond_str} "
        f"| 指数1位: {h['name']}（指数{h['idx_mar']:.0f}）父:{h['sire']} 母父:{h['msire']} "
        f"推定{h['popularity']}番人気"
    )


def build_prompt(data: dict) -> str:
    preview_date = data["preview_date"]
    weekday_map = ["月", "火", "水", "木", "金", "土", "日"]
    date_label = (
        f"{preview_date.year}年{preview_date.month}月{preview_date.day}日"
        f"（{weekday_map[preview_date.weekday()]}）"
    )

    # featured races（タイトルに含まれる注目レース）
    featured = data.get("featured_races", [])
    featured_section = ""
    for i, r in enumerate(featured, 1):
        label = f"[featured_race_{i}] "
        featured_section += _format_race_block(r, label=label) + "\n\n"
    if not featured_section:
        featured_section = "（指定なし）"

    # featured以外の注目レース
    featured_ids = {r["race_id"] for r in featured}
    other_notable = [r for r in data.get("notable_races", []) if r["race_id"] not in featured_ids]
    other_section = "\n\n".join(
        _format_race_block(r) for r in other_notable
    ) or "（なし）"

    # 新馬戦：突出した馬のみ
    maiden_lines = []
    for r in data.get("maiden_races", []):
        top = r.get("top_horses", [])
        if len(top) >= 2 and top[0]["idx_mar"] - top[1]["idx_mar"] >= 10:
            maiden_lines.append(_format_maiden_block(r))
        elif top and top[0]["idx_mar"] >= 80:
            maiden_lines.append(_format_maiden_block(r))
    maiden_str = "\n".join(maiden_lines) if maiden_lines else "（突出した指数の馬なし。このセクションは省略してください）"

    # 全レース概要
    all_lines = []
    for r in data.get("races", []):
        info = r.get("info", {})
        cond = f"{info.get('race_type','')}{info.get('course_len','')}m"
        th = r["top_horses"][0] if r["top_horses"] else None
        mar_str = f"MAR1位:{th['name']}(指数{th['idx_mar']:.0f})" if th else ""
        all_lines.append(
            f"- {r['venue']} R{r['race_num']} {r['race_name']} {cond} {r['entry_count']}頭 {mar_str}"
        )
    all_str = "\n".join(all_lines) if all_lines else "（データなし）"

    return f"""以下のデータをもとに、明日の競馬コラム（前日展望）を書いてください。

## 対象日
{date_label}

## タイトル注目レース（記事で最初に取り上げるレース。featured_race_1 → featured_race_2 の順に展望してください）
{featured_section}

## その他の特別戦・オープン戦
{other_section}

## 新馬戦ピックアップ候補
{maiden_str}

## 全レース一覧（概要）
{all_str}

---
{_ARTICLE_GUIDE}
"""


def generate(
    preview_date: date,
    featured_race_names: list[str] | None = None,
) -> tuple[str, dict]:
    """前日展望記事を生成する

    Args:
        preview_date: 展望する日（土曜 or 日曜）
        featured_race_names: タイトルに含まれる注目レース名リスト（この順に展望）

    Returns:
        (article_markdown_text, data_context)
    """
    data = collect_preview_data(preview_date, featured_race_names)
    prompt = build_prompt(data)
    text = claude_api_client.generate_article_text(_SYSTEM_PROMPT, prompt)
    return text, data
