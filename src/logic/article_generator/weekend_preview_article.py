"""週末重賞展望記事のデータ収集・プロンプト生成・記事生成

木曜 20:00 に実行。土日の重賞レースを対象に、
・レース概要（コース・条件）
・昨年の勝ち馬
・同コースの最近の傾向（人気別勝率など）
・推定人気（スクレイピングで出走馬が取れていた場合）
から、展望記事を生成する。

予想スコアは未計算のため使用しない。
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd

from src.config import paths
from src.managers import race_card_dataset_manager as rcm

# ─── 定数 ───────────────────────────────────────────────────────────────────

_PLACE = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}
_PLACE_DIR = {
    "札幌": "01_sapporo", "函館": "02_hakodate", "福島": "03_fukushima",
    "新潟": "04_nigata", "東京": "05_tokyo", "中山": "06_nakayama",
    "中京": "07_chukyo", "京都": "08_kyoto", "阪神": "09_hanshin",
    "小倉": "10_kokura",
}

_SYSTEM_PROMPT = """あなたは競馬情報サイト「MAR（まーる）」のコラムライターです。

【サイトの特徴】
- LightGBMを使ったAI予想モデル「MAR」が各レースの注目馬を選出
- 的中率重視（MAR-hit）・回収率重視（MAR-val）・バランス型（MAR）の3指数を提供
- 読者は競馬ファンで、毎週土日に競馬を楽しんでいる

【記事スタイル】
- 一人称は「私」または「MAR」
- 週末への期待感を高める、わくわくした文体
- データ・数字は具体的に（四捨五入して読みやすく）
- 断定的な「この馬が勝つ」表現は避け「注目している」「面白そう」等を使う
- 読者への自然な問いかけを盛り込む

【注意事項】
- 木曜時点の情報のため、最終的な騎手変更・回避馬が出る可能性を自然に示す
- 推定人気はあくまで現時点での目安であることを明示する
- 前日段階の予想スコアはまだ未計算のため、その話題には触れない
- 表（テーブル）は一切使わない。数字は文章に自然に組み込む
- レース名は「**京都大賞典**」のように太字にする
- 馬名は「**テーオーロイヤル**」のように太字にする
- Markdownの見出し（## や ###）を使う

【禁止事項】
- 馬券購入の断定的な推奨
- 未確認情報の記述"""

_ARTICLE_GUIDE = """
## 記事構成（合計2,500〜3,000字程度）

### [リード]（150字）
今週末の競馬への期待感から書き出す。開催場、重賞のラインナップなど。

### 注目重賞（1つめ）（800字）
データに「featured_race_1」として指定されたレースを取り上げる。
**必ず含める情報：**
- 開催場・何レース目・レース名・距離・コース条件（芝/ダート）
- このレースの位置づけ・特徴（2〜3行）
- 昨年の勝ち馬（渡されたデータにあれば）と当時のレース振り返り
- 同コース・距離の最近の傾向（人気別勝率、差し有利・逃げ有利など）
- 今年の出走馬で注目している馬（推定人気から）。「この馬が勝つ」ではなく「注目を集めている」程度の表現に留める
- 推定人気上位馬の血統・条件適性を簡潔に触れる

### 注目重賞（2つめ）（700字）
データに「featured_race_2」として指定されたレースを同様に取り上げる。

### その他の重賞・特別戦（300字）
残りの重賞があれば1〜2行ずつ簡潔に。

### 読者へのひとこと（150字）
「今週末はどのレースに期待していますか？」等、読者が答えやすい問いかけで締める。

---
- 構成はデータに応じて自然に調整してよい
- 重賞が1つしかない場合は「注目重賞（1つめ）」だけ書けばよい
- 推定人気データがない場合は「出走馬が出揃い次第、MAR指数でも注目したい」程度に触れる
- 昨年成績がない場合は省略してよい（新設重賞など）
"""


# ─── データ収集ユーティリティ ───────────────────────────────────────────────

def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


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


def _get_entries(race_id: str, race_day: date) -> list[dict]:
    """出走馬エントリーを取得（木曜時点でまだなければ空リスト）"""
    card = rcm.get_race_cards(race_day, race_id)
    if card.empty:
        return []
    entries = []
    for _, r in card.iterrows():
        entries.append({
            "name": str(r.get("馬名", "")),
            "sex_age": str(r.get("性齢", "")),
            "jockey": str(r.get("騎手", "")),
            "trainer": str(r.get("厩舎", "")),
            "sire": str(r.get("peds_0", "")),
            "msire": str(r.get("peds_4", "")),
            "odds": str(r.get("オッズ", "---")),
            "popularity": str(r.get("人気", "?")),
        })
    # 人気順にソート（人気が数値でない場合は末尾へ）
    def _pop_key(e):
        try:
            return int(e["popularity"])
        except (ValueError, TypeError):
            return 999
    entries.sort(key=_pop_key)
    return entries


def _get_past_winner(race_name: str, venue: str, preview_date: date) -> dict | None:
    """前年同名レースの1着馬を取得する（daily_preview_article と同ロジック）"""
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
        (df_r.index.astype(str) == str(target_race_id)) &
        (df_r["着順"] == "1")
    ]
    if winners.empty:
        df_r2 = df_r.reset_index()
        id_col = df_r2.columns[0]
        winners = df_r2[
            (df_r2[id_col].astype(str) == str(target_race_id)) &
            (df_r2["着順"] == "1")
        ]
    if winners.empty:
        return None
    w = winners.iloc[0]
    return {
        "name": str(w.get("馬名", "")),
        "jockey": str(w.get("騎手", "")),
        "odds": str(w.get("単勝", "")),
        "popularity": str(w.get("人気", "")),
        "time": str(w.get("タイム", "")),
        "race_id": target_race_id,
    }


def _get_course_tendency(venue: str, race_type: str, course_len: str,
                         as_of_date: date, n: int = 20) -> dict:
    """同コース・同タイプの最近N勝者から傾向データを計算する

    Returns:
        {
          "sample_n": int,          # 集計対象レース数
          "fav_win_rate": float,     # 1番人気の勝率（%）
          "avg_winner_pop": float,   # 勝ち馬の平均人気
          "pop_dist": str,           # 人気別勝利数のサマリー文字列
        }
    """
    place_dir = _PLACE_DIR.get(venue)
    if not place_dir:
        return {}

    try:
        course_len_int = int(course_len)
    except (ValueError, TypeError):
        return {}

    winners = []
    for yr in range(as_of_date.year, as_of_date.year - 4, -1):
        result_csv = os.path.join(
            paths.DATA_PATH, "race_result", place_dir, f"{yr}_race_results.csv"
        )
        if not os.path.exists(result_csv):
            continue
        df = pd.read_csv(result_csv, dtype=str, index_col=0)
        if "着順" not in df.columns:
            continue
        df_win = df[df["着順"] == "1"].copy()
        if "course_len" in df_win.columns:
            df_win = df_win[df_win["course_len"] == str(course_len_int)]
        if "race_type" in df_win.columns and race_type:
            df_win = df_win[df_win["race_type"] == race_type]
        if "date" in df_win.columns:
            df_win = df_win[df_win["date"].str.contains(str(yr), na=False)]
        winners.extend(df_win.to_dict("records"))
        if len(winners) >= n * 2:
            break

    if not winners:
        return {}

    winners = winners[-n:]
    pops = []
    for w in winners:
        try:
            pops.append(int(w.get("人気", "")))
        except (ValueError, TypeError):
            pass

    if not pops:
        return {}

    fav_wins = sum(1 for p in pops if p == 1)
    avg_pop = sum(pops) / len(pops)
    pop_count: dict[int, int] = {}
    for p in pops:
        pop_count[p] = pop_count.get(p, 0) + 1

    # 人気別サマリー（上位5番人気まで）
    pop_lines = []
    for p in sorted(pop_count):
        if p <= 5:
            pop_lines.append(f"{p}番人気: {pop_count[p]}勝")
    if any(p > 5 for p in pop_count):
        cnt = sum(v for k, v in pop_count.items() if k > 5)
        pop_lines.append(f"6番人気以下: {cnt}勝")

    return {
        "sample_n": len(pops),
        "fav_win_rate": round(fav_wins / len(pops) * 100, 1),
        "avg_winner_pop": round(avg_pop, 1),
        "pop_dist": " / ".join(pop_lines),
    }


# ─── データ収集メイン ─────────────────────────────────────────────────────────

def collect_weekend_preview_data(sat_date: date, sun_date: date | None = None) -> dict:
    """週末重賞展望に必要なデータを収集する

    Returns:
        {
          "sat_date": date,
          "sun_date": date | None,
          "graded_races": [  # G1→G2→G3→重賞順
            {
              "race_id", "race_name", "grade", "venue",
              "race_num", "race_day",
              "info": dict,        # コース・距離等
              "entries": list,     # 出走馬（木曜時点で空の場合も）
              "past_winner": dict|None,
              "tendency": dict,    # コース傾向
            }, ...
          ],
        }
    """
    days = [d for d in [sat_date, sun_date] if d is not None]
    graded_races = []

    for race_day in days:
        df_time = rcm.get_race_time_id_list_df(race_day)
        if df_time.empty:
            continue
        for _, row in df_time.iterrows():
            race_id = str(row["race_id"])
            race_name = str(row.get("race_name", ""))
            grade_csv = row.get("grade")
            grade_str = str(grade_csv) if pd.notna(grade_csv) else ""

            # 重賞のみ対象（G1/G2/G3）
            if grade_str not in ("G1", "G2", "G3"):
                continue

            place_code = race_id[4:6]
            venue = _PLACE.get(place_code, place_code)
            race_num = int(race_id[10:12])

            info = _get_race_info(race_id)
            # race_infoに grade が入っていればそちらを優先
            if not grade_str and info.get("grade"):
                grade_str = info["grade"]

            entries = _get_entries(race_id, race_day)
            past_winner = _get_past_winner(race_name, venue, race_day)
            tendency = _get_course_tendency(
                venue,
                info.get("race_type", ""),
                info.get("course_len", ""),
                sat_date,
            )

            graded_races.append({
                "race_id": race_id,
                "race_name": race_name,
                "grade": grade_str,
                "venue": venue,
                "race_num": race_num,
                "race_day": race_day,
                "info": info,
                "entries": entries,
                "past_winner": past_winner,
                "tendency": tendency,
            })

    # G1 > G2 > G3 順にソート（同グレードは race_num 降順）
    grade_order = {"G1": 0, "G2": 1, "G3": 2}
    graded_races.sort(key=lambda r: (grade_order.get(r["grade"], 9), -r["race_num"]))

    return {
        "sat_date": sat_date,
        "sun_date": sun_date,
        "graded_races": graded_races,
    }


# ─── プロンプト構築 ───────────────────────────────────────────────────────────

def _format_race_section(race: dict, label: str) -> str:
    info = race["info"]
    rt = info.get("race_type", "")
    cl = info.get("course_len", "")
    cond = f"{rt}{cl}m" if rt and cl else ""

    lines = [f"[{label}]"]
    lines.append(f"レース名: {race['race_name']} ({race['grade']})")
    lines.append(f"開催: {race['venue']} {race['race_num']}R / {race['race_day']}")
    if cond:
        lines.append(f"コース: {cond}")

    # 昨年の勝ち馬
    pw = race.get("past_winner")
    if pw:
        lines.append(
            f"昨年の勝ち馬: {pw['name']} "
            f"（騎手: {pw['jockey']}, {pw['popularity']}番人気, "
            f"単勝{pw['odds']}倍, タイム{pw['time']}）"
        )
    else:
        lines.append("昨年の勝ち馬: データなし")

    # コース傾向
    tend = race.get("tendency", {})
    if tend:
        lines.append(
            f"同コース傾向（直近{tend['sample_n']}勝）: "
            f"1番人気勝率 {tend['fav_win_rate']}%, "
            f"勝ち馬平均人気 {tend['avg_winner_pop']}番人気"
        )
        lines.append(f"  人気別内訳: {tend['pop_dist']}")
    else:
        lines.append("同コース傾向: データなし")

    # 出走馬（推定人気順）
    entries = race.get("entries", [])
    if entries:
        lines.append(f"出走馬（推定人気順・{len(entries)}頭）:")
        for e in entries[:8]:  # 上位8頭
            sire_info = f"（父: {e['sire']}）" if e["sire"] else ""
            lines.append(
                f"  {e['popularity']}番人気 {e['name']}{sire_info} "
                f"騎手: {e['jockey']} / オッズ: {e['odds']}倍"
            )
        if len(entries) > 8:
            lines.append(f"  ※他 {len(entries) - 8}頭")
    else:
        lines.append("出走馬: 木曜時点では未取得（当日までに更新予定）")

    return "\n".join(lines)


def build_prompt(data: dict) -> str:
    graded = data["graded_races"]
    sat = data["sat_date"]
    sun = data["sun_date"]

    date_range = f"{sat.month}/{sat.day}（土）"
    if sun:
        date_range += f"〜{sun.month}/{sun.day}（日）"

    lines = [
        f"# 週末重賞展望データ（{date_range}）",
        f"重賞レース数: {len(graded)}",
        "",
    ]

    featured = graded[:2]
    rest = graded[2:]

    for i, race in enumerate(featured, 1):
        lines.append(_format_race_section(race, f"featured_race_{i}"))
        lines.append("")

    if rest:
        lines.append("[その他の重賞]")
        for race in rest:
            info = race["info"]
            rt = info.get("race_type", "")
            cl = info.get("course_len", "")
            cond = f"{rt}{cl}m" if rt and cl else ""
            lines.append(
                f"- {race['race_name']} ({race['grade']}) "
                f"{race['venue']} {race['race_num']}R {cond}"
            )
        lines.append("")

    lines.append("---")
    lines.append(_ARTICLE_GUIDE)
    return "\n".join(lines)


# ─── 記事生成 ─────────────────────────────────────────────────────────────────

def generate(sat_date: date, sun_date: date | None = None) -> tuple[str, dict]:
    """週末重賞展望記事を生成する

    Returns:
        (article_text, data)
    """
    from src.logic.article_generator.claude_api_client import generate_article_text

    data = collect_weekend_preview_data(sat_date, sun_date)
    if not data["graded_races"]:
        return "", data

    prompt = build_prompt(data)
    article_text = generate_article_text(_SYSTEM_PROMPT, prompt)
    return article_text, data
