"""傾向分析ページのHTML生成

public_html/trend/ 以下に以下を生成する:
  - YYYYMMDD.html       日次短評ページ（当日夜に生成）
  - weekly_YYYYMMDD.html 週次振り返りページ（水曜に生成、SatのYYYYMMDDを使用）
  - index.html          一覧ページ（最新エントリ20件）
"""

import json
import os
import re
from datetime import date, datetime

from src.config import paths
from src.logic.html_generator.site_nav_html import (
    AD_SLOT_IN_CONTENT_1,
    AD_SLOT_IN_CONTENT_2,
    SITE_URL,
    _trend_venue_names,
    ad_unit_html,
    adsense_script_html,
    breadcrumb_html,
    ga4_script_html,
    meta_tags_html,
    sidebar_html,
    site_footer_html,
    site_nav_html,
)
TREND_DIR = os.path.join(paths.PUBLIC_HTML_PATH, "trend")
os.makedirs(TREND_DIR, exist_ok=True)

_CSS_VER = 1


def _write_html(path: str, html: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def _build_sidebar(current_file: str = "") -> str:
    """傾向ページの右サイドバー（一覧へ戻るリンクのみ）"""
    return sidebar_html(
        [],
        up_link=("傾向分析一覧", "index.html"),
    )


def _css_version() -> int:
    css_path = os.path.join(paths.PUBLIC_HTML_ASSETS_PATH, "css", "styles.css")
    try:
        return int(os.path.getmtime(css_path))
    except OSError:
        return _CSS_VER


def _entry_label(date_key: str, is_weekly: bool = False) -> str:
    y, m, d = date_key[:4], date_key[4:6], date_key[6:8]
    if is_weekly:
        return f"{y}年{m}月{d}日（土）週次振り返り"
    weekday = date(int(y), int(m), int(d)).weekday()
    wd = ["月", "火", "水", "木", "金", "土", "日"][weekday]
    return f"{y}年{m}月{d}日（{wd}）傾向短評"


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


def _inline_md(text: str) -> str:
    """インラインMarkdown（太字・イタリック）をHTMLに変換する"""
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


def _md_table_to_html(block: str) -> str:
    """Markdownテーブルブロックを <table> に変換する"""
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    rows = []
    for line in lines:
        if re.match(r"^\|[-:| ]+\|$", line):
            continue  # 区切り行スキップ
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return ""
    thead = "".join(f"<th>{_inline_md(c)}</th>" for c in rows[0])
    tbody_rows = ""
    for row in rows[1:]:
        tbody_rows += "<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in row) + "</tr>"
    return (
        '<div class="trend-md-table-wrap">'
        f'<table class="trend-md-table"><thead><tr>{thead}</tr></thead>'
        f"<tbody>{tbody_rows}</tbody></table></div>"
    )


def _text_to_html(text: str) -> str:
    """生成テキストをHTML段落に変換する（Markdown見出し・リスト・テーブル・太字対応）"""
    text = text.strip()
    paragraphs = []
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            continue
        # 水平線
        if re.match(r"^-{3,}$|^\*{3,}$", block):
            continue  # 区切り線は非表示
        # Markdown見出し（#〜####）
        if block.startswith("#### "):
            paragraphs.append(f'<h5 class="trend-section-title">{_inline_md(block[5:].strip())}</h5>')
        elif block.startswith("### "):
            paragraphs.append(f'<h4 class="trend-section-title">{_inline_md(block[4:].strip())}</h4>')
        elif block.startswith("## "):
            paragraphs.append(f'<h3 class="trend-section-title">{_inline_md(block[3:].strip())}</h3>')
        elif block.startswith("# "):
            paragraphs.append(f'<h2 class="trend-section-title">{_inline_md(block[2:].strip())}</h2>')
        elif block.startswith("■"):
            content = block.lstrip("■").strip()
            paragraphs.append(f'<h2 class="trend-section-title">■ {_inline_md(content)}</h2>')
        # Markdownテーブル（|で始まる複数行）
        elif "|" in block and block.startswith("|"):
            paragraphs.append(_md_table_to_html(block))
        # リスト（- または * で始まる行の連続）
        elif re.match(r"^[-*] ", block):
            items = []
            for line in block.splitlines():
                line = line.strip()
                if re.match(r"^[-*] ", line):
                    items.append(f"<li>{_inline_md(line[2:].strip())}</li>")
                elif line:
                    items[-1] = items[-1][:-5] + "<br>" + _inline_md(line) + "</li>" if items else f"<li>{_inline_md(line)}</li>"
            paragraphs.append(f'<ul class="trend-md-list">{"".join(items)}</ul>')
        else:
            inner = _inline_md(block.replace("\n", "<br>"))
            paragraphs.append(f"<p>{inner}</p>")
    return "\n".join(paragraphs)


def _up3f_speed_label(up3f_val: float, place: str, race_type: str,
                       baselines: dict | None) -> str:
    """上り3Fの速度ラベルを返す。

    baselines が与えられた場合は競馬場・コース種別の過去平均と比較し、
    なければ固定閾値にフォールバックする。
    """
    baseline = (baselines or {}).get(place, {}).get(race_type)
    if baseline is not None:
        diff = up3f_val - baseline
        if diff <= -0.8:
            spd = "速"
        elif diff <= -0.3:
            spd = "やや速"
        elif diff >= 0.8:
            spd = "遅"
        elif diff >= 0.3:
            spd = "やや遅"
        else:
            spd = "標準"
        sign = "+" if diff >= 0 else ""
        return f'{spd}<span class="cs-baseline">avg{baseline:.1f}比{sign}{diff:.1f}s</span>'
    # フォールバック: 固定閾値
    if race_type == "芝":
        spd = "速" if up3f_val <= 34.5 else ("やや速" if up3f_val <= 35.2 else ("標準" if up3f_val <= 35.8 else "遅"))
    else:
        spd = "速" if up3f_val <= 38.0 else ("標準" if up3f_val <= 39.5 else "遅")
    return spd


_POP_SEG_COLORS = [
    "pop-fav1", "pop-fav2", "pop-fav3",
    "pop-mid", "pop-longshot", "pop-upset",
]
_POP_LEGEND_LABELS = [
    "1番人気", "2番人気", "3番人気",
    "4〜6番", "7〜9番", "10番〜",
]


def _stacked_bar_row(bar_dist: list, label: str, total: int,
                      is_venue: bool = False) -> str:
    """1行分のスタック棒を生成する"""
    row_cls = "pop-row-venue" if is_venue else "pop-row-main"
    segs = ""
    for i, d in enumerate(bar_dist):
        if d["count"] == 0:
            continue
        color_cls = _POP_SEG_COLORS[i] if i < len(_POP_SEG_COLORS) else "pop-mid"
        segs += (
            f'<div class="pop-seg {color_cls}" style="flex:{d["count"]}" '
            f'title="{d["label"]} {d["count"]}R">'
            f'<span class="pop-seg-count">{d["count"]}</span>'
            f'</div>'
        )
    return (
        f'<div class="pop-stacked-row {row_cls}">'
        f'<span class="pop-stacked-label">{label}</span>'
        f'<div class="pop-stacked-bar">{segs}</div>'
        f'<span class="pop-stacked-total">{total}R</span>'
        f'</div>\n'
    )


def _winner_pop_chart_html(stats: dict) -> str:
    """勝ち馬の人気分布をスタック棒グラフで生成する（全体＋開催場別）"""
    wpd = stats.get("winner_pop_dist", {})
    if not wpd or not wpd.get("dist"):
        return ""

    dist = wpd["dist"]
    by_place = wpd.get("by_place", {})
    payout = wpd.get("payout", {})
    n = wpd.get("race_count", 0)

    rows_html = _stacked_bar_row(dist, "全体", n)
    if by_place:
        rows_html += '<div class="pop-venue-divider">開催別</div>\n'
        for place_name, pd_data in by_place.items():
            rows_html += _stacked_bar_row(
                pd_data["dist"], place_name, pd_data["race_count"], is_venue=True
            )

    payout_html = ""
    if payout:
        avg = payout.get("avg_payout", 0)
        mx = payout.get("max_payout", 0)
        payout_html = (
            f'<div class="trend-payout-summary">'
            f'<span class="payout-item"><span class="payout-label">平均払戻</span>'
            f'<span class="payout-val">{avg:,}円</span></span>'
            f'<span class="payout-item"><span class="payout-label">最高払戻</span>'
            f'<span class="payout-val">{mx:,}円</span></span>'
            f'</div>'
        )

    legend_html = "".join(
        f'<span class="pop-leg-item">'
        f'<span class="pop-leg-swatch {_POP_SEG_COLORS[i]}"></span>{lbl}</span>'
        for i, lbl in enumerate(_POP_LEGEND_LABELS)
    )

    return (
        f'<div class="trend-pop-dist">'
        f'<div class="pop-dist-header">'
        f'<span class="trend-pop-dist-title">勝ち馬の人気分布</span>'
        f'{payout_html}'
        f'</div>'
        f'{rows_html}'
        f'<div class="pop-legend">{legend_html}</div>'
        f'</div>'
    )


def _top3_pop_chart_html(stats: dict) -> str:
    """3着内馬の人気分布チャート（複勝・三連複払い戻し付き）"""
    wpd = stats.get("top3_pop_dist", {})
    if not wpd or not wpd.get("dist"):
        return ""

    dist = wpd["dist"]
    by_place = wpd.get("by_place", {})
    payout = wpd.get("payout", {})
    n = wpd.get("race_count", 0)
    h = wpd.get("horse_count", 0)

    rows_html = _stacked_bar_row(dist, "全体", n)
    if by_place:
        rows_html += '<div class="pop-venue-divider">開催別</div>\n'
        for place_name, pd_data in by_place.items():
            rows_html += _stacked_bar_row(
                pd_data["dist"], place_name, pd_data.get("race_count", 0), is_venue=True
            )

    payout_html = ""
    if payout:
        items = []
        if "fukusho_avg" in payout:
            items.append(
                f'<span class="payout-item">'
                f'<span class="payout-label">複勝平均</span>'
                f'<span class="payout-val">{payout["fukusho_avg"]:,}円</span></span>'
                f'<span class="payout-item">'
                f'<span class="payout-label">複勝最高</span>'
                f'<span class="payout-val">{payout["fukusho_max"]:,}円</span></span>'
            )
        if "sanrenpuku_avg" in payout:
            items.append(
                f'<span class="payout-item">'
                f'<span class="payout-label">三連複平均</span>'
                f'<span class="payout-val">{payout["sanrenpuku_avg"]:,}円</span></span>'
                f'<span class="payout-item">'
                f'<span class="payout-label">三連複最高</span>'
                f'<span class="payout-val">{payout["sanrenpuku_max"]:,}円</span></span>'
            )
        if items:
            payout_html = f'<div class="trend-payout-summary">{"".join(items)}</div>'

    legend_html = "".join(
        f'<span class="pop-leg-item">'
        f'<span class="pop-leg-swatch {_POP_SEG_COLORS[i]}"></span>{lbl}</span>'
        for i, lbl in enumerate(_POP_LEGEND_LABELS)
    )

    return (
        f'<div class="trend-pop-dist">'
        f'<div class="pop-dist-header">'
        f'<span class="trend-pop-dist-title">3着内馬の人気分布</span>'
        f'{payout_html}'
        f'</div>'
        f'{rows_html}'
        f'<div class="pop-legend">{legend_html}</div>'
        f'</div>'
    )


def _stats_summary_html(stats: dict, baselines: dict | None = None) -> str:
    """開催場ごと・芝ダート別のデータサマリHTMLを生成する"""
    if not stats or "error" in stats:
        return ""

    GROUND_CLS = {"良": "ground-good", "稍重": "ground-soft", "重": "ground-heavy", "不良": "ground-bad"}

    # 開催場ごとのカード（芝/ダート別内訳付き）
    venue_cards = []
    for place, info in stats.get("ground_by_place", {}).items():
        ground = info.get("ground_state", "不明")
        weather = info.get("weather", "不明")
        rc = info.get("race_count", 0)
        g_cls = GROUND_CLS.get(ground, "")

        # 芝/ダート別行
        course_rows = []
        for rt in ["芝", "ダート"]:
            cs = info.get("course_stats", {}).get(rt)
            if not cs:
                continue
            up = f"{cs['up3f']}秒" if cs.get("up3f") else "—"
            if cs.get("up3f"):
                spd_label = _up3f_speed_label(cs["up3f"], place, rt, baselines)
                up = f"{up}<small>（{spd_label}）</small>"
            upset_n = cs.get("upset_count", 0)
            r_c = cs.get("race_count", 0)
            upset_str = f"{upset_n}/{r_c}R" if r_c else "—"
            course_rows.append(
                f'<tr><td class="cs-type">{rt}</td>'
                f'<td class="cs-up3f">{up}</td>'
                f'<td class="cs-upset">{upset_str}</td></tr>'
            )

        course_table = ""
        if course_rows:
            course_table = (
                '<table class="trend-course-table">'
                '<thead><tr><th></th><th>上り3F</th><th>波乱</th></tr></thead>'
                f'<tbody>{"".join(course_rows)}</tbody>'
                '</table>'
            )

        venue_cards.append(f"""<div class="trend-venue-card2">
  <div class="trend-venue-header">
    <span class="trend-venue-name">{place}</span>
    <span class="trend-ground-badge {g_cls}">{ground}</span>
    <span class="trend-venue-meta">天気:{weather} / {rc}R</span>
  </div>
  {course_table}
</div>""")

    # 全体荒れ度
    upset = stats.get("upset", {})
    upset_label = upset.get("label", "不明")
    upset_badge_cls = _ROUGHNESS_BADGE_CLS.get(upset_label, "roughness-neutral")
    total_r = stats.get("race_count", upset.get("race_count", 0))
    high_r = upset.get("high_odds_count", 0)

    # 場別荒れ傾向バッジ（全体と同じフォーマット）
    upset_by_place = stats.get("upset_by_place", {})
    by_place_items = "".join(
        f'<span class="roughness-item">'
        f'<span class="roughness-place">{place}</span>'
        f'<span class="roughness-label {_ROUGHNESS_BADGE_CLS.get(data["label"], "roughness-neutral")}">{data["label"]}</span>'
        f'</span>'
        for place, data in upset_by_place.items()
    )

    pop_chart = _winner_pop_chart_html(stats)
    top3_chart = _top3_pop_chart_html(stats)

    return f"""<div class="trend-stats-block">
  <div class="trend-venues2">{"".join(venue_cards)}</div>
  <div class="trend-roughness-summary">
    <div class="trend-roughness-overall">
      <span class="trend-roughness-section-label">全体</span>
      <span class="roughness-label {upset_badge_cls}">{upset_label}</span>
      <span class="trend-roughness-note">（単勝10倍超 {high_r}/{total_r}R）</span>
    </div>
    <div class="trend-roughness-row trend-roughness-venues">
      <span class="trend-roughness-section-label">開催別</span>
      {by_place_items}
    </div>
  </div>
  {pop_chart}
  {top3_chart}
</div>"""


def make_daily_trend_page(target_date: date, stats: dict, comment_text: str) -> None:
    """日次傾向短評ページを生成する

    Args:
        target_date: 対象日付
        stats: trend_analyzer.get_day_stats() の返り値
        comment_text: trend_text_generator.generate_daily_comment() の返り値
    """
    date_key = target_date.strftime("%Y%m%d")
    filename = f"{date_key}.html"
    out_path = os.path.join(TREND_DIR, filename)

    page_url = f"{SITE_URL}/trend/{filename}"
    title_date = f"{target_date.year}年{target_date.month:02d}月{target_date.day:02d}日"
    weekday = ["月", "火", "水", "木", "金", "土", "日"][target_date.weekday()]
    page_title = f"{title_date}（{weekday}）傾向短評 | MAR"
    description = f"{title_date}のJRA競馬傾向分析。馬場状態・荒れ度・AI予想成績をまとめた短評です。"

    css_ver = _css_version()
    stats_html = _stats_summary_html(stats, stats.get("up3f_baselines"))
    comment_html = _text_to_html(comment_text)

    venue_names = "・".join(stats.get("ground_by_place", {}).keys())

    html = f"""{_head_html(page_title, description, page_url, css_ver)}
<body>
{site_nav_html(base_path="../", current_path=f"trend/{filename}")}
<div class="content-wrapper">
  <main class="main-content">
    {breadcrumb_html([("傾向分析", "index.html"), (title_date, "")])}
    <article class="trend-article">
      <header class="trend-header">
        <div class="trend-date-badge">{title_date}（{weekday}）</div>
        <h1 class="trend-title">傾向短評 — {venue_names}</h1>
        <p class="trend-generated-at">更新日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M")}</p>
      </header>
      {stats_html}
      {ad_unit_html(AD_SLOT_IN_CONTENT_1)}
      <div class="trend-text-body">
        {comment_html}
      </div>
      {ad_unit_html(AD_SLOT_IN_CONTENT_2)}
    </article>
  </main>
  {_build_sidebar(filename)}
</div>
{site_footer_html()}
</body>
</html>"""

    _write_html(out_path, html)

    # 荒れ傾向をJSONキャッシュに保存（インデックス表示用）
    _save_roughness_to_json(date_key, stats)

    _update_index()


def _save_roughness_to_json(date_key: str, stats: dict) -> None:
    """日次JSONに roughness キーを保存（なければスキップ）"""
    roughness = {place: data["label"] for place, data in stats.get("upset_by_place", {}).items()}
    if not roughness:
        return
    json_path = os.path.join(paths.DATA_PATH, "trend", "daily", f"{date_key}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            saved = {"date": date_key}
    else:
        saved = {"date": date_key}
    saved["roughness"] = roughness
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False, indent=2)


def make_weekly_trend_page(sat_date: date, sun_date: date,
                            week_stats: dict, comment_text: str) -> None:
    """週次振り返りページを生成する

    Args:
        sat_date: 土曜の日付
        sun_date: 日曜の日付
        week_stats: trend_analyzer.get_week_stats() の返り値
        comment_text: trend_text_generator.generate_weekly_comment() の返り値
    """
    date_key = sat_date.strftime("%Y%m%d")
    filename = f"weekly_{date_key}.html"
    out_path = os.path.join(TREND_DIR, filename)

    page_url = f"{SITE_URL}/trend/{filename}"
    sat_label = f"{sat_date.year}年{sat_date.month:02d}月{sat_date.day:02d}日"
    sun_label = f"{sun_date.month:02d}月{sun_date.day:02d}日"
    page_title = f"{sat_label}〜{sun_label} 週次傾向振り返り | MAR"
    description = f"{sat_label}〜{sun_label}のJRA競馬週次傾向振り返り。展開傾向・馬場変化・来週末予測。"

    css_ver = _css_version()
    baselines = week_stats.get("up3f_baselines")
    sat_stats_html = _stats_summary_html(week_stats.get("sat", {}), baselines)
    sun_stats_html = _stats_summary_html(week_stats.get("sun", {}), baselines)
    comment_html = _text_to_html(comment_text)

    # 展開傾向バッジ
    pace = week_stats.get("combined", {}).get("pace_bias", {})
    pace_html = ""
    if pace:
        front_pct = pace.get("front_rate", 0) * 100
        closer_pct = pace.get("closer_rate", 0) * 100
        total = pace.get("valid_races", 0)
        if front_pct > 40:
            pace_label = "前有利"
            pace_cls = "pace-front"
        elif closer_pct > 30:
            pace_label = "後ろ有利"
            pace_cls = "pace-closer"
        else:
            pace_label = "互角"
            pace_cls = "pace-even"
        pace_html = f"""<div class="trend-pace-block">
  <span class="trend-indicator-label">展開傾向（土日合算 {total}R）</span>
  <span class="trend-pace-badge {pace_cls}">{pace_label}</span>
  <span class="trend-indicator-sub">前残り {front_pct:.1f}% / 差し追込 {closer_pct:.1f}%</span>
</div>"""

    html = f"""{_head_html(page_title, description, page_url, css_ver)}
<body>
{site_nav_html(base_path="../", current_path=f"trend/{filename}")}
<div class="content-wrapper">
  <main class="main-content">
    {breadcrumb_html([("傾向分析", "index.html"), (f"{sat_label}週次", "")])}
    <article class="trend-article">
      <header class="trend-header">
        <div class="trend-date-badge weekly-badge">週次振り返り</div>
        <h1 class="trend-title">{sat_label}〜{sun_label}</h1>
        <p class="trend-generated-at">更新日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M")}</p>
      </header>

      <section class="trend-day-section">
        <h2 class="trend-day-title">土曜（{sat_label}）</h2>
        {sat_stats_html}
      </section>

      <section class="trend-day-section">
        <h2 class="trend-day-title">日曜（{sun_label}）</h2>
        {sun_stats_html}
      </section>

      {pace_html}

      {ad_unit_html(AD_SLOT_IN_CONTENT_1)}

      <div class="trend-text-body">
        {comment_html}
      </div>

      {ad_unit_html(AD_SLOT_IN_CONTENT_2)}
    </article>
  </main>
  {_build_sidebar(filename)}
</div>
{site_footer_html()}
</body>
</html>"""

    _write_html(out_path, html)
    _update_index()


def _roughness_from_json(date_key: str) -> dict:
    """日次JSONの roughness キーを {場名: ラベル} で返す"""
    json_path = os.path.join(paths.DATA_PATH, "trend", "daily", f"{date_key}.json")
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return {}
    return saved.get("roughness", {})


_ROUGHNESS_BADGE_CLS = {
    "固め": "roughness-firm",
    "やや固め": "roughness-somewhat-firm",
    "やや荒れ": "roughness-somewhat-rough",
    "荒れ": "roughness-rough",
}


def _roughness_badges_html(roughness: dict, small: bool = False) -> str:
    """荒れ傾向バッジHTMLを生成する（短評ページとインデックス共通）"""
    if not roughness:
        return ""
    row_cls = "trend-roughness-row roughness-small" if small else "trend-roughness-row"
    items = []
    for place, label in roughness.items():
        badge_cls = _ROUGHNESS_BADGE_CLS.get(label, "roughness-neutral")
        items.append(
            f'<span class="roughness-item">'
            f'<span class="roughness-place">{place}</span>'
            f'<span class="roughness-label {badge_cls}">{label}</span>'
            f'</span>'
        )
    # インデックス内では <a> の中に入るため span タグを使用
    tag = "span" if small else "div"
    return f'<{tag} class="{row_cls}">{"".join(items)}</{tag}>'


def _update_index() -> None:
    """public_html/trend/index.html を再生成する（月別グループ表示）"""
    from datetime import timedelta
    from itertools import groupby as _groupby

    rows = []
    for fname in os.listdir(TREND_DIR):
        if not fname.endswith(".html") or fname == "index.html":
            continue
        # diary セクションに移動した review_ ファイルはスキップ
        if fname.startswith("review_"):
            continue
        is_weekly = fname.startswith("weekly_")
        date_key = fname.replace("weekly_", "").replace(".html", "")
        if len(date_key) != 8:
            continue
        try:
            y, m, d = int(date_key[:4]), int(date_key[4:6]), int(date_key[6:8])
            sat = date(y, m, d)
        except (ValueError, IndexError):
            continue
        venues = _trend_venue_names(os.path.join(TREND_DIR, fname))
        venue_suffix = f"（{venues}）" if venues else ""
        if is_weekly:
            sun = sat + timedelta(days=1)
            sort_day, sort_type = sun.day, 1
            short_label = f"{m}/{d}〜{sun.month}/{sun.day} 週次振り返り{venue_suffix}"
            roughness = {}
        else:
            weekday_ja = ["月", "火", "水", "木", "金", "土", "日"][sat.weekday()]
            sort_day, sort_type = d, 0
            short_label = f"{m}/{d}（{weekday_ja}）"
            roughness = _roughness_from_json(date_key)
        rows.append((y, m, sort_day, sort_type, fname, is_weekly, short_label, roughness))

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

    groups = []
    for (y, m_key), grp in _groupby(rows, key=lambda r: (r[0], r[1])):
        groups.append((y, m_key, list(grp)))
    groups.sort(key=lambda g: (g[0], g[1]), reverse=True)

    sections_html = ""
    if not groups:
        sections_html = '<p class="trend-index-empty">エントリがまだありません。</p>'
    else:
        for y, m_key, grp_rows in groups:
            badge_html_list = []
            for _, _, _, _, fname, is_weekly, short_label, roughness in reversed(grp_rows):
                badge_cls = "weekly" if is_weekly else "daily"
                badge_text = "週次" if is_weekly else "短評"
                roughness_html = _roughness_badges_html(roughness, small=True) if not is_weekly else ""
                label_text = short_label if is_weekly else f"{short_label}短評"
                badge_html_list.append(
                    f'<li class="trend-index-entry">'
                    f'<span class="entry-badge {badge_cls}">{badge_text}</span>'
                    f'<span class="trend-entry-wrap">'
                    f'<a href="{fname}">{label_text}</a>'
                    f'{roughness_html}'
                    f'</span>'
                    f'</li>'
                )
            entries_inner = "\n".join(badge_html_list)
            sections_html += (
                f'<section class="trend-month-section">'
                f'<h2 class="trend-month-title">{y}年{m_key}月</h2>'
                f'<ul class="trend-index-list">{entries_inner}</ul>'
                f'</section>\n'
            )

    css_ver = _css_version()
    page_url = f"{SITE_URL}/trend/index.html"
    page_title = "傾向分析 | MAR"
    description = "MAR(まーる)が分析するJRA競馬の馬場傾向・荒れ度・展開傾向の日次・週次レポート。"

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
{site_nav_html(base_path="../", current_path="trend/index.html")}
<div class="content-wrapper">
  <main class="main-content">
    {breadcrumb_html([("傾向分析", "")])}
    <div class="page-header">
      <h1>傾向分析</h1>
      <div class="page-desc">
        <p>レース終了後に各開催場の傾向（荒れ度・差し決着の多さなど）を分析した日次短評と、土日まとめの週次振り返りです。</p>
        <p>荒れ度指標は<strong>単勝10倍超の勝ち馬の割合</strong>をもとに4段階で示します。</p>
        <ul class="trend-index-legend">
          <li><span class="roughness-label roughness-firm">固め</span> 10倍超0〜15%未満。1〜3番人気が素直に決まりやすい週。</li>
          <li><span class="roughness-label roughness-somewhat-firm">やや固め</span> 10倍超15〜30%未満。基本は人気馬中心だが、中穴がときおり絡む週。</li>
          <li><span class="roughness-label roughness-somewhat-rough">やや荒れ</span> 10倍超30〜50%未満。中穴・大穴が台頭しやすい週。</li>
          <li><span class="roughness-label roughness-rough">荒れ</span> 10倍超50%以上。波乱続出の週。</li>
        </ul>
        <p class="page-desc-sub">傾向バッジはエントリーのタイトル横に場ごとに表示されます。先週・前日の傾向と今週を比べて、傾向の推移を把握するのにご活用ください。</p>
      </div>
    </div>
    <div class="trend-index-groups">
{sections_html}
    </div>
  </main>
  {_build_sidebar()}
</div>
{site_footer_html()}
</body>
</html>"""

    out_path = os.path.join(TREND_DIR, "index.html")
    _write_html(out_path, html)
