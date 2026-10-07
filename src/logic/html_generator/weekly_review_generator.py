"""MARの競馬コラム（②）のHTMLページ生成

public_html/diary/YYYYMMDD.html（公開日付）として出力する。
傾向分析とは独立したセクション。
"""

import json
import os
from datetime import date, datetime
from itertools import groupby as _groupby

from src.config import paths
from src.logic.html_generator.site_nav_html import (
    AD_SLOT_IN_CONTENT_1,
    AD_SLOT_IN_CONTENT_2,
    SITE_URL,
    ad_unit_html,
    adsense_script_html,
    breadcrumb_html,
    ga4_script_html,
    meta_tags_html,
    site_footer_html,
    site_nav_html,
)
from src.logic.html_generator.trend_page_generator import (
    _css_version,
    _text_to_html,
    _write_html,
)

DIARY_DIR = os.path.join(paths.PUBLIC_HTML_PATH, "diary")
DIARY_DATA_DIR = os.path.join(paths.DATA_PATH, "diary")
os.makedirs(DIARY_DIR, exist_ok=True)
os.makedirs(DIARY_DATA_DIR, exist_ok=True)

_WEEKDAY = ["月", "火", "水", "木", "金", "土", "日"]

# article_type → バッジCSSクラス
_TYPE_BADGE_CLS = {
    "振り返り": "review",
    "週末展望": "preview-weekend",
    "前日展望": "preview-tomorrow",
}


def _weekday_str(d: date) -> str:
    return _WEEKDAY[d.weekday()]


def _perf_summary_html(perf: dict, article_type: str = "振り返り") -> str:
    """AI成績サマリーのデータカード"""
    win = perf.get("win", {})
    place = perf.get("place", {})
    trio = perf.get("trio_box", {})
    n = win.get("n", 0)
    if n == 0:
        return ""

    def _rate_cls(r: float) -> str:
        if r >= 100:
            return "rate-good"
        if r >= 80:
            return "rate-neutral"
        return "rate-bad"

    card_title = "先週末のMAR成績" if article_type == "振り返り" else "今週末のMAR成績"

    items = [
        ("単勝", win.get("hit_rate", 0), win.get("return_rate", 0)),
        ("複勝", place.get("hit_rate", 0), place.get("return_rate", 0)),
        ("3連複BOX", trio.get("hit_rate", 0), trio.get("return_rate", 0)),
    ]
    item_html = ""
    for label, hr, rr in items:
        roi_cls = _rate_cls(rr)
        item_html += f"""<div class="review-perf-item">
  <div class="review-perf-type">{label}</div>
  <div class="review-perf-stats">
    <div class="review-perf-stat">
      <span class="review-perf-stat-val">{hr:.1f}<small>%</small></span>
      <span class="review-perf-stat-label">的中率</span>
    </div>
    <div class="review-perf-stat {roi_cls}">
      <span class="review-perf-stat-val">{rr:.1f}<small>%</small></span>
      <span class="review-perf-stat-label">回収率</span>
    </div>
  </div>
</div>"""

    return f"""<div class="review-perf-card">
  <div class="review-perf-header">
    <span class="review-perf-title">{card_title}</span>
    <span class="review-perf-n">対象 {n}R</span>
  </div>
  <div class="review-perf-grid">{item_html}</div>
</div>"""


def _head_html(title: str, description: str, url: str, css_ver: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {adsense_script_html()}
  {ga4_script_html()}
  {meta_tags_html(title, description, url)}
  <link rel="stylesheet" href="../assets/css/styles.css?v={css_ver}">
  <link rel="canonical" href="{url}">
</head>"""


def make_weekly_review_page(
    sat_date: date,
    sun_date: date | None,
    article_text: str,
    data: dict,
    article_type: str = "振り返り",
    main_race: str = "",
    pub_date: date | None = None,
    venues: list[str] | None = None,
) -> str:
    """MARの競馬コラムページを生成する

    Args:
        sat_date: 土曜日（サブタイトル・date_range の基準）
        sun_date: 日曜日（タイトル表示用。前日展望の場合は None）
        article_text: Markdown記事テキスト
        data: collect_weekend_data() / collect_preview_data() の返り値
        article_type: "振り返り" / "週末展望" / "前日展望"
        main_race: メインレース名（タイトルに入れる、省略可）
        pub_date: 公開日（省略時は sat_date）
        venues: 開催場リスト（例: ["東京", "京都"]）。サブタイトルに追加

    Returns:
        生成したHTMLファイルのパス
    """
    pub_d = pub_date or sat_date
    date_key = pub_d.strftime("%Y%m%d")
    filename = f"{date_key}.html"
    out_path = os.path.join(DIARY_DIR, filename)
    pub_label = f"{pub_d.month}/{pub_d.day}（{_weekday_str(pub_d)}）"

    sat_label = f"{sat_date.month}/{sat_date.day}（{_weekday_str(sat_date)}）"
    if sun_date:
        sun_label = f"・{sun_date.month}/{sun_date.day}（{_weekday_str(sun_date)}）"
    else:
        sun_label = ""
    date_range = f"{sat_label}{sun_label}"
    venue_str = f"｜{'・'.join(venues)}" if venues else ""
    subtitle_text = f"{date_range}{venue_str}"

    # タイプ別の説明文プレフィックス
    _desc_prefix = {
        "振り返り": f"先週末（{date_range}）の振り返り。",
        "週末展望": f"今週末（{date_range}）の注目レース展望。",
        "前日展望": f"明日（{date_range}）の注目レース展望。",
    }
    desc_prefix = _desc_prefix.get(article_type, f"{date_range}の{article_type}。")

    # タイトル：公開日 MARの競馬コラム（メインレース名：タイプ）
    if main_race:
        page_title = f"{pub_label} MARの競馬コラム（{main_race}：{article_type}）| MAR"
        h1_text = f"{main_race}：{article_type}"
        description = f"{desc_prefix}{main_race}のレース結果・見どころをお届けします。"
    else:
        page_title = f"{pub_label} MARの競馬コラム：{article_type} | MAR"
        h1_text = article_type
        description = f"{desc_prefix}今週末の競馬をお届けします。"

    page_url = f"{SITE_URL}/diary/{filename}"

    # メタデータをJSONに保存（indexページ再生成に使用）
    meta = {
        "date_key": date_key,
        "sat": sat_date.isoformat(),
        "sun": sun_date.isoformat() if sun_date else None,
        "pub_date": pub_d.isoformat(),
        "article_type": article_type,
        "main_race": main_race,
        "venues": venues or [],
        "page_title": page_title,
    }
    meta_path = os.path.join(DIARY_DATA_DIR, f"{date_key}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    css_ver = _css_version()
    perf_html = _perf_summary_html(data.get("perf", {}), article_type)
    article_html = _text_to_html(article_text)

    badge_cls = _TYPE_BADGE_CLS.get(article_type, "review")

    html = f"""{_head_html(page_title, description, page_url, css_ver)}
<body>
{site_nav_html(base_path="../", current_path=f"diary/{filename}")}
<main>
  {breadcrumb_html([("MARの競馬コラム", "diary/index.html"), (h1_text, None)], base_path="../")}
  <article class="trend-article">
    <header class="trend-header review-header">
      <div class="trend-date-badge {badge_cls}-badge">{article_type}</div>
      <h1 class="trend-title">{h1_text}</h1>
      <p class="trend-subtitle">{subtitle_text}</p>
      <p class="trend-generated-at">公開日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M")}</p>
    </header>

    {perf_html}

    {ad_unit_html(AD_SLOT_IN_CONTENT_1)}

    <div class="trend-text-body review-text-body">
      {article_html}
    </div>

    {ad_unit_html(AD_SLOT_IN_CONTENT_2)}

    <p class="diary-back-link"><a href="index.html">← MARの競馬コラム一覧へ戻る</a></p>
  </article>
</main>
{site_footer_html()}
</body>
</html>"""

    _write_html(out_path, html)
    _update_diary_index()
    print(f"MARの競馬コラムを生成しました: {out_path}")
    return out_path


def _update_diary_index() -> None:
    """public_html/diary/index.html を再生成する（月別グループ表示）"""
    # JSONメタデータを読み込む
    rows = []
    for fname in os.listdir(DIARY_DATA_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(DIARY_DATA_DIR, fname), encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        date_key = meta.get("date_key", fname.replace(".json", ""))
        if len(date_key) != 8:
            continue
        try:
            y, m, d = int(date_key[:4]), int(date_key[4:6]), int(date_key[6:8])
        except (ValueError, IndexError):
            continue

        sat = date.fromisoformat(meta["sat"]) if meta.get("sat") else date(y, m, d)
        sun = date.fromisoformat(meta["sun"]) if meta.get("sun") else None
        pub_d = date.fromisoformat(meta["pub_date"]) if meta.get("pub_date") else sat
        article_type = meta.get("article_type", "振り返り")
        main_race = meta.get("main_race", "")

        pub_label = f"{pub_d.month}/{pub_d.day}（{_weekday_str(pub_d)}）"
        date_range = f"{sat.month}/{sat.day}"
        if sun:
            date_range += f"〜{sun.month}/{sun.day}"

        if main_race:
            short_label = f"{pub_label} {main_race}：{article_type}"
        else:
            short_label = f"{pub_label} {date_range} {article_type}"

        rows.append((y, m, d, f"{date_key}.html", short_label, article_type))

    # JSONがなく HTMLだけある古いファイルをフォールバックで拾う
    existing_keys = {r[3].replace(".html", "") for r in rows}
    for fname in os.listdir(DIARY_DIR):
        if not fname.endswith(".html") or fname == "index.html":
            continue
        date_key = fname.replace(".html", "")
        if len(date_key) != 8 or date_key in existing_keys:
            continue
        try:
            y, m, d = int(date_key[:4]), int(date_key[4:6]), int(date_key[6:8])
            sat = date(y, m, d)
            sun_d = d + 1
            sun = date(y, m, sun_d) if sun_d <= 28 else sat
        except (ValueError, IndexError):
            continue
        short_label = f"{m}/{d}〜{sun.month}/{sun.day} 振り返り"
        rows.append((y, m, d, fname, short_label, "振り返り"))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    groups = []
    for (y, m_key), grp in _groupby(rows, key=lambda r: (r[0], r[1])):
        groups.append((y, m_key, list(grp)))
    groups.sort(key=lambda g: (g[0], g[1]), reverse=True)

    css_ver = _css_version()
    page_title = "MARの競馬コラム | MAR"
    page_url = f"{SITE_URL}/diary/index.html"
    description = "毎週末のレース展望と振り返りをお届けするコラムです。週末の注目レース展望（木曜）、前日の見どころ（金・土）、週末の振り返り（火曜）の3本立てで更新します。"

    sections_html = ""
    if not groups:
        sections_html = '<p class="trend-index-empty">エントリがまだありません。</p>'
    else:
        for y, m_key, grp_rows in groups:
            badge_html_list = []
            for _, _, _, fname, short_label, article_type in reversed(grp_rows):
                badge_cls = _TYPE_BADGE_CLS.get(article_type, "review")
                badge_html_list.append(
                    f'<li class="trend-index-entry">'
                    f'<span class="entry-badge {badge_cls}">{article_type}</span>'
                    f'<a href="{fname}">{short_label}</a>'
                    f'</li>'
                )
            entries_inner = "\n".join(badge_html_list)
            sections_html += (
                f'<section class="trend-month-section">'
                f'<h2 class="trend-month-title">{y}年{m_key}月</h2>'
                f'<ul class="trend-index-list">{entries_inner}</ul>'
                f'</section>\n'
            )

    from src.logic.html_generator.site_nav_html import (
        adsense_script_html, ga4_script_html, meta_tags_html,
        site_nav_html, breadcrumb_html, site_footer_html,
    )

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{page_title}</title>
  {adsense_script_html()}
  {ga4_script_html()}
  {meta_tags_html(page_title, description, page_url)}
  <link rel="stylesheet" href="../assets/css/styles.css?v={css_ver}">
  <link rel="canonical" href="{page_url}">
</head>
<body>
{site_nav_html(base_path="../", current_path="diary/index.html")}
<div class="content-wrapper">
  <main class="main-content">
    {breadcrumb_html([("MARの競馬コラム", None)], base_path="../")}
    <div class="trend-index-page">
      <header class="trend-index-header">
        <h1 class="trend-index-title">MARの競馬コラム</h1>
        <p class="trend-index-desc">
          毎週末のレース展望と振り返りをお届けするコラムです。<br>
          週末の注目レース展望（木曜）・前日の見どころ（金・土）・週末の振り返り（火曜）の3本立てで更新します。
        </p>
      </header>
      {sections_html}
    </div>
  </main>
</div>
{site_footer_html()}
</body>
</html>"""

    index_path = os.path.join(DIARY_DIR, "index.html")
    _write_html(index_path, html)
