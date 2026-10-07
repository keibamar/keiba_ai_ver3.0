"""週末競馬日記（②）のHTMLページ生成

public_html/diary/YYYYMMDD.html（土曜日付）として出力する。
傾向分析とは独立したセクション。
"""

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
    sidebar_html,
    site_footer_html,
    site_nav_html,
)
from src.logic.html_generator.trend_page_generator import (
    _css_version,
    _text_to_html,
    _write_html,
)

DIARY_DIR = os.path.join(paths.PUBLIC_HTML_PATH, "diary")
os.makedirs(DIARY_DIR, exist_ok=True)


def _build_sidebar() -> str:
    return sidebar_html(
        [],
        up_link=("週末競馬日記一覧", "index.html"),
    )


def _perf_summary_html(perf: dict) -> str:
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

    rows = [
        ("単勝", win.get("hit_rate", 0), win.get("return_rate", 0)),
        ("複勝", place.get("hit_rate", 0), place.get("return_rate", 0)),
        ("3連複BOX", trio.get("hit_rate", 0), trio.get("return_rate", 0)),
    ]
    table_rows = "".join(
        f'<tr><th>{label}</th><td>{hr:.1f}%</td>'
        f'<td class="{_rate_cls(rr)}">{rr:.1f}%</td></tr>'
        for label, hr, rr in rows
    )
    return f"""<div class="review-perf-card">
  <div class="review-perf-title">今週末のMAR成績 <span class="review-perf-n">（{n}レース）</span></div>
  <table class="review-perf-table">
    <thead><tr><th>券種</th><th>的中率</th><th>回収率</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
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
) -> str:
    """週末競馬日記ページを生成する

    Args:
        sat_date: 土曜日（ファイル名・URL の基準）
        sun_date: 日曜日（タイトル表示用）
        article_text: Claude API が生成したMarkdown記事テキスト
        data: collect_weekend_data() の返り値

    Returns:
        生成したHTMLファイルのパス
    """
    date_key = sat_date.strftime("%Y%m%d")
    filename = f"{date_key}.html"
    out_path = os.path.join(DIARY_DIR, filename)

    sat_label = f"{sat_date.year}年{sat_date.month:02d}月{sat_date.day:02d}日"
    sun_label = (
        f"〜{sun_date.month:02d}月{sun_date.day:02d}日"
        if sun_date else ""
    )
    page_title = f"{sat_label}{sun_label} 週末競馬日記 | MAR"
    description = (
        f"{sat_label}{sun_label}の週末競馬日記。"
        "MARのAI予想結果と注目レース・注目馬の振り返り。"
        "今週の馬券トピックと読者への問いかけ。"
    )
    page_url = f"{SITE_URL}/diary/{filename}"

    css_ver = _css_version()
    perf_html = _perf_summary_html(data.get("perf", {}))
    article_html = _text_to_html(article_text)

    html = f"""{_head_html(page_title, description, page_url, css_ver)}
<body>
{site_nav_html(base_path="../", current_path=f"diary/{filename}")}
<div class="content-wrapper">
  <main class="main-content">
    {breadcrumb_html([("週末競馬日記", "index.html"), (f"{sat_label}{sun_label}", "")])}
    <article class="trend-article">
      <header class="trend-header review-header">
        <div class="trend-date-badge review-badge">週末競馬日記</div>
        <h1 class="trend-title">{sat_label}{sun_label}</h1>
        <p class="trend-generated-at">公開日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M")}</p>
      </header>

      {perf_html}

      {ad_unit_html(AD_SLOT_IN_CONTENT_1)}

      <div class="trend-text-body review-text-body">
        {article_html}
      </div>

      {ad_unit_html(AD_SLOT_IN_CONTENT_2)}
    </article>
  </main>
  {_build_sidebar()}
</div>
{site_footer_html()}
</body>
</html>"""

    _write_html(out_path, html)
    _update_diary_index()
    print(f"週末競馬日記を生成しました: {out_path}")
    return out_path


def _update_diary_index() -> None:
    """public_html/diary/index.html を再生成する（月別グループ表示）"""
    rows = []
    for fname in os.listdir(DIARY_DIR):
        if not fname.endswith(".html") or fname == "index.html":
            continue
        date_key = fname.replace(".html", "")
        if len(date_key) != 8:
            continue
        try:
            y, m, d = int(date_key[:4]), int(date_key[4:6]), int(date_key[6:8])
            sat = date(y, m, d)
        except (ValueError, IndexError):
            continue
        sun = date(y, m, d + 1) if d < 28 else sat  # 大まかな日曜日付
        short_label = f"{m}/{d}〜{sun.month}/{sun.day} 週末競馬日記"
        rows.append((y, m, d, fname, short_label))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    groups = []
    for (y, m_key), grp in _groupby(rows, key=lambda r: (r[0], r[1])):
        groups.append((y, m_key, list(grp)))
    groups.sort(key=lambda g: (g[0], g[1]), reverse=True)

    css_ver = _css_version()
    page_title = "週末競馬日記 | MAR"
    page_url = f"{SITE_URL}/diary/index.html"
    description = "MAR週末競馬日記。毎週火曜更新。土日の競馬をAI予想の結果とともに振り返ります。"

    sections_html = ""
    if not groups:
        sections_html = '<p class="trend-index-empty">エントリがまだありません。</p>'
    else:
        for y, m_key, grp_rows in groups:
            badge_html_list = []
            for _, _, _, fname, short_label in reversed(grp_rows):
                badge_html_list.append(
                    f'<li class="trend-index-entry">'
                    f'<span class="entry-badge review">日記</span>'
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
    {breadcrumb_html([("週末競馬日記", "")])}
    <div class="trend-index-page">
      <header class="trend-index-header">
        <h1 class="trend-index-title">週末競馬日記</h1>
        <p class="trend-index-desc">
          毎週火曜更新。土日の競馬をAI予想（MAR）の成績とともに振り返り、
          今週のハイライトレースや注目馬をお届けします。
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
