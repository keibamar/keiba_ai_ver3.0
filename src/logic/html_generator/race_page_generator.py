"""レース個別ページ（Forge: HTMLFactory）のHTML生成

旧 web/src/generators/race_pages.py からの移植。HTMLテンプレート・解析ロジックは
変更せず、データ取得のみ新アーキテクチャのManager層に切り替えている。
"""

import html
import os
import re
from datetime import date, datetime, timedelta

import pandas as pd

from src.config.constants import NAME_LIST, PLACE_LIST, RANK_COLORS, WAKU_COLORS

# AI予想Rank用の配色（人気と同系色だが薄め、視覚的に区別しやすくする）
PRED_RANK_COLORS = {"1": "#FFF9C4", "2": "#B2EBF2", "3": "#FFE0B2"}
from src.logic.html_generator import affiliate_html, horse_report_generator
from src.logic.prediction.race_prediction_engine import score_to_index
from src.managers import score_calibration_manager as scm
from src.logic.html_generator.site_nav_html import (
    AD_SLOT_IN_CONTENT_1,
    AD_SLOT_IN_CONTENT_2,
    AD_SLOT_IN_CONTENT_3,
    SITE_URL,
    ad_unit_html,
    adsense_script_html,
    breadcrumb_html,
    ga4_script_html,
    meta_tags_html,
    site_footer_html,
    site_nav_html,
)
from src.managers import (
    html_manager,
    peds_results_dataset_manager,
    race_card_dataset_manager,
    race_info_dataset_manager,
    race_result_dataset_manager,
    race_schedule_dataset_manager,
)
from src.utils.format_data import format_date, merge_rank_score


def read_race_csv(date_str, target_id):
    """出馬表+score/rankを読み込んで必要列を返す。失敗時はNoneを返す"""
    race_day = datetime.strptime(date_str, "%Y%m%d").date()
    df = race_card_dataset_manager.get_race_cards(race_day, target_id)
    if df.empty:
        print(f"ℹ️ [スキップ/エラーではありません] レースカード未生成: race_day={date_str}, target_id={target_id}")
        return None
    cols = [
        "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "馬体重(増減)", "オッズ", "人気",
        "idx_hitrate", "rank_hitrate", "idx_value", "rank_value", "idx_mar", "rank_mar",
    ]
    existing = list(dict.fromkeys([c for c in cols if c in df.columns]))
    return df[existing]


def get_result_table(date_str, place_id, target_id):
    df_race = race_result_dataset_manager.get_race_id_result(target_id)
    if df_race.empty:
        print(f"ℹ️ [スキップ/エラーではありません] レース結果データなし（未確定）: {target_id}")
        return pd.DataFrame()
    return df_race


def get_returns_table(date_str, place_id, target_id):
    df_race = race_info_dataset_manager.get_race_return_csv_for_race(target_id)
    if df_race.empty:
        print(f"ℹ️ [スキップ/エラーではありません] 配当結果データなし（未確定）: {target_id}")
    return df_race


def get_race_info(year, place_id, target_id):
    df_info = race_card_dataset_manager.get_race_info_csv(target_id)
    if not df_info.empty:
        race_type = str(df_info.iloc[0].get("race_type", "") or "")
        course_len_value = df_info.iloc[0].get("course_len", "")
        course_len = None
        if pd.notna(course_len_value):
            course_len_str = str(course_len_value).strip()
            if course_len_str != "":
                try:
                    course_len = int(float(course_len_str))
                except Exception:
                    course_len = None
        ground_state = str(df_info.iloc[0].get("ground_state", "") or "")
        race_class = str(df_info.iloc[0].get("class", "") or "")

        # --- クラス表記を統一（全角数字 → 半角数字）---
        trans_table = str.maketrans("０１２３４５６７８９", "0123456789")
        race_class = race_class.translate(trans_table)
        return race_type, course_len, ground_state, race_class
    else:
        print(f"ℹ️ [スキップ/エラーではありません] レース情報未蓄積: {target_id}")
        return None, None, None, None


def _top_n_by_score(df, score_col, n=5):
    """指定したscore列の上位n頭の(馬番, 馬名)リストを返す（降順。データが無ければ空リスト）"""
    if score_col not in df.columns or "馬番" not in df.columns or "馬名" not in df.columns:
        return []
    scores = pd.to_numeric(df[score_col], errors="coerce")
    sub = df.assign(_score=scores).dropna(subset=["_score"]).sort_values("_score", ascending=False)
    return [(int(row["馬番"]), row["馬名"]) for _, row in sub.head(n).iterrows()]


def _sub_model_picks_column_html(label, emoji, picks):
    if not picks:
        return f"""<div class="ai-sub-model-picks-col">
        <p class="ai-sub-model-picks-label">{emoji} {label}</p>
        <p class="ai-sub-model-picks-empty">データなし</p>
      </div>"""
    items = "".join(f"<li>{num}番 {html.escape(str(name))}</li>" for num, name in picks)
    return f"""<div class="ai-sub-model-picks-col">
        <p class="ai-sub-model-picks-label">{emoji} {label}</p>
        <ol class="ai-sub-model-picks-list">{items}</ol>
      </div>"""


def build_ai_pick_summary_html(df):
    """的中率重視・回収率重視モデルのTOP5（現在は非表示）"""
    return ""


def _weight_change_style(body_str):
    """馬体重(増減)の文字列（例: "472(-4)"）から、増減が大きい馬を強調する文字色styleを返す

    +10kg以上の増加は赤、-10kg以上の減少は青の文字色で強調する（コンディション
    変化が大きい馬として一覧で目立たせるため。背景は塗らない）。
    増減が読み取れない場合は空文字列。
    """
    match = re.search(r"\(([+-]?\d+)\)", str(body_str))
    if not match:
        return ""
    diff = int(match.group(1))
    if diff >= 10:
        return 'color:red; font-weight:bold;'
    if diff <= -10:
        return 'color:blue; font-weight:bold;'
    return ""


def _seirei_style(seirei):
    """性齢文字列から文字色styleを返す（牝→赤、セン→青、牡→デフォルト黒）"""
    s = str(seirei)
    if s.startswith("牝"):
        return "color:#c62828; font-weight:bold;"
    if s.startswith("セン"):
        return "color:#1565c0; font-weight:bold;"
    return ""


def _odds_cell_html(odds_raw, tag="td"):
    """単勝オッズの表示用HTMLを返す（色付け付き）

    10倍未満: 赤（人気馬）、100倍以上: 青（大穴）、その他: 黒。
    出馬表・結果表で共通して使用する。
    """
    odds_str = str(odds_raw).strip() if odds_raw is not None else ""
    try:
        val = float(odds_str)
        if val < 10:
            style = "color:#c62828; font-weight:bold;"
        elif val >= 100:
            style = "color:#1f4fd6; font-weight:bold;"
        else:
            style = "color:#333;"
    except (ValueError, TypeError):
        style = "color:#333;"
    return f'<{tag} style="{style}">{odds_str}</{tag}>'


def _odds_update_label_html(date_str, target_id, is_confirmed):
    """オッズ・人気の更新時刻ラベルHTMLを返す

    is_confirmed=True（レース結果確定済み）のとき「確定オッズ」表示。
    未確定のときはrace_card CSVのファイル更新時刻を表示する。
    """
    from src.config import paths
    if is_confirmed:
        return '<p class="odds-update-label odds-update-label--confirmed">確定オッズ</p>'
    card_path = os.path.join(paths.RACE_CARD_DATA_PATH, date_str, f"{target_id}.csv")
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(card_path))
        label = mtime.strftime("%m/%d %H:%M").lstrip("0")
        return f'<p class="odds-update-label">オッズ・人気 {label} 時点</p>'
    except Exception:
        return ""


def _ai_index_cell_html(ai_index):
    """AI指数の表示用TD HTMLを返す（色付け・ツールチップ付き）

    ≤39: 青（低評価）、40-59: 標準、60-69: 赤（高評価）、70以上: 赤+強調背景。
    ai-score-col クラスで列全体に薄い背景と左ボーダーを付ける（CSSで定義）。
    """
    if ai_index is None or ai_index == "":
        return '<td class="ai-score-col"></td>'

    # 色付け（4段階）。70以上は背景色もインラインで上書き（クラス背景より優先）
    if ai_index >= 70:
        style = "color:#c62828; font-weight:bold; background-color:#ffebee;"
    elif ai_index >= 60:
        style = "color:#c62828; font-weight:bold;"
    elif ai_index <= 39:
        style = "color:#1f4fd6; font-weight:bold;"
    else:
        style = "color:#333;"

    # ツールチップ
    band = scm.find_band(ai_index)
    if band:
        n = int(band["n"]) if band.get("n") else 0
        n_str = f"{n:,}頭の実績 " if n else ""
        tooltip = (
            f"AI指数 {int(band['band_min'])}〜{int(band['band_max'])}点帯"
            f"（{n_str}単勝率{float(band['win_rate']):.1f}% / 複勝率{float(band['place_rate']):.1f}%）"
            f" ※絶対評価スコア（コース・距離別の能力を0〜100点で表示）"
        )
    else:
        tooltip = "AI指数: コース・距離別の絶対的な能力スコア（0〜100点、高いほど高評価）"

    td_attrs = f'class="ai-score-col" title="{html.escape(tooltip)}" style="{style}"'

    return f"<td {td_attrs}>{ai_index}</td>"


def _mm_index_cell_html(ai_index, rank, css_class, is_main=False):
    """マルチモデル指数の表示用TD HTML

    is_main=True（MAR列）はメイン指数として大きめフォント・強いランク色付け。
    is_main=False（的中率/回収率）はサブ指数として小さめ・薄いランク色付け。
    data-sort 属性に数値を持たせてJSソートを確実に動作させる。
    """
    if ai_index is None or ai_index == "" or (isinstance(ai_index, float) and pd.isna(ai_index)):
        return f'<td class="{css_class}" data-sort="0"></td>'
    try:
        val = float(ai_index)
    except (ValueError, TypeError):
        return f'<td class="{css_class}" data-sort="0"></td>'
    try:
        rank_int = int(rank) if rank != "" and pd.notna(rank) else None
    except (ValueError, TypeError):
        rank_int = None

    # ランク別背景色
    if rank_int == 1:
        bg = "#FFD700" if is_main else "#FFF9C4"   # 金 / 薄金
    elif rank_int == 2:
        bg = "#B0BEC5" if is_main else "#E0F7FA"   # 銀 / 薄水色
    elif rank_int == 3:
        bg = "#FFAB76" if is_main else "#FFE0B2"   # 銅 / 薄橙
    else:
        bg = ""

    # 指数値の文字色
    if val >= 70:
        fg = "#c62828"
    elif val >= 60:
        fg = "#b71c1c"
    elif val <= 39:
        fg = "#1f4fd6"
    else:
        fg = "#333"

    font_size = "1.15em" if is_main else "0.9em"
    fw = "bold" if is_main or (rank_int and rank_int <= 3) else "normal"
    bg_css = f"background-color:{bg};" if bg else ""
    style = f"{bg_css}color:{fg};font-size:{font_size};font-weight:{fw};"

    rank_str = (
        f'<sup style="font-size:0.72em;margin-left:2px;color:#555;">({rank_int})</sup>'
        if rank_int else ""
    )
    return f'<td class="{css_class}" data-sort="{val:.2f}" style="{style}">{val:.1f}{rank_str}</td>'


def build_table_race_cards(df):
    """メインの出走表（csv側）から HTML の行文字列を作成"""
    if df is None or df.empty:
        return ""
    rows = ""
    for idx, (_, row) in enumerate(df.iterrows()):
        # 安全に値を取り出す
        waku = int(row['枠']) if '枠' in row and pd.notna(row['枠']) else ""
        umaban = int(row['馬番']) if '馬番' in row and pd.notna(row['馬番']) else ""
        name = row.get('馬名', '')
        # 性齢・斤量・騎手・馬体重は確定前はNaNのことがあり、そのままf-stringに
        # 渡すと文字どおり"nan"が表示されてしまうため、未確定時は空欄にする
        seirei = row.get('性齢', '') if pd.notna(row.get('性齢', '')) else ''
        kinryo = row.get('斤量', '') if pd.notna(row.get('斤量', '')) else ''
        jockey = row.get('騎手', '') if pd.notna(row.get('騎手', '')) else ''
        body = row.get('馬体重(増減)', '') if pd.notna(row.get('馬体重(増減)', '')) else ''
        # 単勝オッズ（発走前に確定。未確定時は「---.-」等）
        odds_raw = row.get('オッズ', '')

        # 人気は発走15〜20分前のレースカード再取得時にスクレイピングされる。
        # オッズ未確定時はnetkeiba側が「**」等のプレースホルダーを返すため、
        # 数値として読めない場合は未確定として「-」表示にする
        popularity_raw = row.get('人気', '')
        try:
            popularity = str(int(float(popularity_raw)))
        except (ValueError, TypeError):
            popularity = "-"
        # --- 人気上位3頭の色付け（金・水色・サーモン = 鮮やか系） ---
        pop_color = RANK_COLORS.get(popularity, "#ffffff")
        pop_style = f'background-color:{pop_color};'
        # --- 馬体重の増減が大きい馬を強調 ---
        weight_style = _weight_change_style(body)
        # --- 性齢の色付け（牝→赤、セン→青、牡→デフォルト） ---
        seirei_style = _seirei_style(seirei)

        # --- 枠順背景色（結果表・配当表と同じ配色で一覧性を揃える） ---
        waku_color = WAKU_COLORS.get(str(waku), "#ffffff")
        waku_style = f'background-color:{waku_color}; color:{"#fff" if str(waku) in ["2", "3", "4", "7"] else "#000"};'

        # 馬名をクリック可能にする（詳細レポートへのリンク）
        unique_id = f"horse_report_{idx}_{umaban}"
        name_html = f'<a href="javascript:void(0);" onclick="scrollToReport(\'{unique_id}\')" style="color: blue; text-decoration: underline; cursor: pointer;">{html.escape(str(name))}</a>'

        # 3戦略マルチモデル指数
        idx_hitrate  = row.get('idx_hitrate', '')
        rank_hitrate = row.get('rank_hitrate', '')
        idx_value    = row.get('idx_value', '')
        rank_value   = row.get('rank_value', '')
        idx_mar      = row.get('idx_mar', '')
        rank_mar     = row.get('rank_mar', '')

        rows += f"""
        <tr>
          <td style="{waku_style}">{waku}</td>
          <td style="{waku_style}">{umaban}</td>
          <td>{name_html}</td>
          <td style="{seirei_style}">{seirei}</td>
          <td>{kinryo}</td>
          <td>{jockey}</td>
          <td style="{weight_style}">{body}</td>
          <td style="{pop_style}">{popularity}</td>
          {_odds_cell_html(odds_raw)}
          {_mm_index_cell_html(idx_hitrate, rank_hitrate, "mm-hitrate-col", is_main=False)}
          {_mm_index_cell_html(idx_value,   rank_value,   "mm-value-col",   is_main=False)}
          {_mm_index_cell_html(idx_mar,     rank_mar,     "mm-mar-col",     is_main=True)}
        </tr>
        """
    return rows


def _race_card_breadcrumb_items(date_str, date_display, place_id, target_id, race_name):
    """出馬表ページの右側タブ・パンくず用の階層（同日の開催場＋選択中の開催場の他レース）を返す

    レースカレンダー→開催日→開催場（同日の他場も兄弟として並べる）→レース
    （同じ開催場の他レースも兄弟として並べる）という階層にする。他の開催場を
    選んだ場合は、その場の同じレース番号のページへ直接遷移する（無ければ
    その場の最初のレースへフォールバックする）。
    """
    race_num = int(str(target_id)[-2:])
    place_name = NAME_LIST[place_id - 1]
    place_key = PLACE_LIST[place_id - 1]
    self_path = f"races/{date_str}/{place_key}R{race_num}.html"

    race_day = datetime.strptime(date_str, "%Y%m%d")
    df_info = race_card_dataset_manager.get_race_time_id_list_df(race_day)

    venue_siblings = [(place_name, self_path)]
    race_siblings = [(f"{race_num}R {race_name}", None)]

    if not df_info.empty:
        ids = df_info["race_id"].astype(str)
        place_ids_today = sorted({int(rid[4:6]) for rid in ids})

        venue_siblings = []
        for pid in place_ids_today:
            pid_name = NAME_LIST[pid - 1]
            pid_key = PLACE_LIST[pid - 1]
            if pid == place_id:
                venue_siblings.append((pid_name, self_path))
                continue
            pid_ids = [rid for rid in ids if int(rid[4:6]) == pid]
            same_num = [rid for rid in pid_ids if int(rid[-2:]) == race_num]
            target_rid = same_num[0] if same_num else min(pid_ids, key=lambda rid: int(rid[-2:]))
            venue_siblings.append((pid_name, f"races/{date_str}/{pid_key}R{int(target_rid[-2:])}.html"))

        df_place = df_info[ids.str.startswith(str(target_id)[:10])].sort_values("race_id")
        race_siblings = []
        for _, row in df_place.iterrows():
            rid = str(row["race_id"])
            rnum = int(rid[-2:])
            label = f"{rnum}R {row['race_name']}"
            race_siblings.append((label, None if rnum == race_num else f"races/{date_str}/{place_key}R{rnum}.html"))
        if not race_siblings:
            race_siblings = [(f"{race_num}R {race_name}", None)]

    return [
        ("レースカレンダー", "races/index.html"),
        (date_display, f"races/{date_str}/index.html"),
        (place_name, self_path, venue_siblings),
        (f"{race_num}R {race_name}", None, race_siblings),
    ]


def build_html_content(date_str, date_display, place_id, race_num, race_name, race_time, target_id, table_rows, run_time_info, weight_info, peds_info, pops_info, frames_info, recent_html, result_table_html, payout_table_html, pick_summary_html="", odds_update_label="", course_link_html="", weather_info_html=""):
    """HTMLテンプレートを返す"""
    race_time_display = f"{race_time[:2]}:{race_time[2:]}" if race_time else ""
    place_name = NAME_LIST[place_id - 1]
    breadcrumb_items = _race_card_breadcrumb_items(date_str, date_display, place_id, target_id, race_name)
    site_nav = site_nav_html(base_path="../../", breadcrumb_items=breadcrumb_items)
    breadcrumb = breadcrumb_html(breadcrumb_items, base_path="../../")
    footer = site_footer_html(base_path="../../")
    adsense = adsense_script_html()
    ga4 = ga4_script_html()
    place_key = PLACE_LIST[place_id - 1]
    meta_tags = meta_tags_html(
        f"{date_display} {place_name}競馬場 第{race_num}R {race_name}",
        f"{date_display} {place_name}競馬場 第{race_num}R {race_name}の出馬表・AI予想データ。",
        f"{SITE_URL}/races/{date_str}/{place_key}R{race_num}.html",
    )
    noindex_tag = '<meta name="robots" content="noindex, follow">' if race_num <= 10 else ""
    # 広告は意味のある区切りに限定して配置する（出馬表の後・配当結果の後・
    # コース別データの後の3箇所。ページ全体で多すぎる印象にならないようにするため）。
    ad_unit_1 = ad_unit_html(AD_SLOT_IN_CONTENT_1)
    ad_unit_2 = ad_unit_html(AD_SLOT_IN_CONTENT_2)
    ad_unit_3 = ad_unit_html(AD_SLOT_IN_CONTENT_3)
    # 配当結果を見終えたタイミングで、補足コンテンツとして書籍紹介を1つだけ置く
    # （広告の手前に挟むことで、広告色を強めずに済む）。
    book_html = affiliate_html.daily_book_recommendation_html()
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {meta_tags}
  {noindex_tag}
  {adsense}
  {ga4}
  <title>{date_display} {place_name}競馬場 第{race_num}R {race_name}</title>
  <link rel="stylesheet" href="../../assets/css/styles.css">
  <link rel="icon" type="image/svg+xml" href="../../assets/favicon.svg">
  <style>
    body {{
      font-family: sans-serif;
      margin: 20px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px;
      text-align: center;
    }}
    th {{
      background-color: #f2f2f2;
      cursor: pointer;
    }}
    .waku-1 {{ background-color: white; }}
    .waku-2 {{ background-color: black; color: white; }}
    .waku-3 {{ background-color: red; color: white; }}
    .waku-4 {{ background-color: blue; color: white; }}
    .waku-5 {{ background-color: yellow; }}
    .waku-6 {{ background-color: green; color: white; }}
    .waku-7 {{ background-color: orange; }}
    .waku-8 {{ background-color: pink; }}
    .rank-1 {{ background-color: yellow; }}
    .rank-2 {{ background-color: lightblue; }}
    .rank-3 {{ background-color: orange; }}
    .score-high {{ color: red; }}
    .score-low {{ color: blue; }}
    .score-verylow {{ color: darkblue; }}

    /* AI指数・Rank列（MARオリジナル予想情報）をボルドー系で区別 */
    .ai-score-col {{
      border-left: 3px solid #7a2438;
    }}
    .ai-rank-col {{
      border-right: 3px solid #7a2438;
    }}
    td.ai-score-col, td.ai-rank-col {{
      font-size: 1.1em;
      font-weight: bold;
    }}
    tbody tr td.ai-score-col,
    tbody tr td.ai-rank-col {{
      background-color: #fbeef0;
    }}
    tbody tr:nth-child(even) td {{
      background-color: #f5f5f5;
    }}
    tbody tr:nth-child(even) td.ai-score-col,
    tbody tr:nth-child(even) td.ai-rank-col {{
      background-color: #f3d9de;
    }}
    th.ai-score-col, th.ai-rank-col {{
      background-color: #7a2438;
      color: white;
    }}

    /* 3戦略マルチモデル指数列 */
    /* サブ指数（的中率・回収率）: 控えめな左ボーダーのみ */
    .mm-hitrate-col {{
      border-left: 3px solid #1565c0;
    }}
    .mm-value-col {{
      border-left: 2px solid #2e7d32;
    }}
    /* メイン指数（MAR）: 両側ボーダーで囲って強調 */
    .mm-mar-col {{
      border-left: 3px solid #6a1b9a;
      border-right: 3px solid #6a1b9a;
    }}
    /* サブ指数の既定セル背景（ランク色で上書きされる場合あり） */
    tbody tr td.mm-hitrate-col {{
      background-color: #f0f4ff;
    }}
    tbody tr td.mm-value-col {{
      background-color: #f0fff4;
    }}
    tbody tr:nth-child(even) td.mm-hitrate-col {{
      background-color: #e3ecff;
    }}
    tbody tr:nth-child(even) td.mm-value-col {{
      background-color: #e3f5e8;
    }}
    /* MAR既定セル背景: やや濃いめ */
    tbody tr td.mm-mar-col {{
      background-color: #f5edff;
    }}
    tbody tr:nth-child(even) td.mm-mar-col {{
      background-color: #ead9ff;
    }}
    /* ヘッダー色 */
    th.mm-hitrate-col {{
      background-color: #1565c0;
      color: white;
      font-size: 0.85em;
    }}
    th.mm-value-col {{
      background-color: #2e7d32;
      color: white;
      font-size: 0.85em;
    }}
    th.mm-mar-col {{
      background-color: #6a1b9a;
      color: white;
      font-size: 1.0em;
      font-weight: bold;
    }}

    /* コースリンク（レース名横）- 芝は緑、ダートは茶 */
    .course-link {{
      font-size: 0.75em;
      font-weight: bold;
      text-decoration: none;
      margin-left: 10px;
      padding: 3px 8px;
      border-radius: 4px;
      vertical-align: middle;
    }}
    .course-link.turf {{
      color: #2e7d32;
      border: 1px solid #81c784;
      background-color: #e8f5e9;
    }}
    .course-link.turf:hover {{
      background-color: #c8e6c9;
    }}
    .course-link.dirt {{
      color: #8d6e3a;
      border: 1px solid #bcaaa4;
      background-color: #efebe9;
    }}
    .course-link.dirt:hover {{
      background-color: #d7ccc8;
    }}

    /* レースヘッダー */
    .race-page-title {{
      font-size: 1.2em;
      margin-bottom: 6px;
    }}
    .race-page-time {{
      font-size: 1.15em;
      font-weight: bold;
      margin: 4px 0;
      color: #333;
    }}
    .race-page-info {{
      font-size: 1.05em;
      color: #555;
      margin: 4px 0 12px;
    }}
    #payoutTable td.num {{
      text-align: right;
      padding-right: 10px;
      white-space: nowrap;
    }}

    /* ========== コース別データセクション（折りたたみ） ========== */
    .course-data-section {{
      margin-top: 20px;
      border: 1px solid #ddd;
      border-radius: 5px;
      background-color: #f9f9f9;
    }}
    .course-data-header {{
      padding: 12px;
      background-color: #007bff;
      color: white;
      cursor: pointer;
      user-select: none;
      font-weight: bold;
      border-radius: 5px 5px 0 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .course-data-header:hover {{
      background-color: #0056b3;
    }}
    .course-data-toggle {{
      font-size: 18px;
      transition: transform 0.3s;
    }}
    .course-data-toggle.open {{
      transform: rotate(180deg);
    }}
    .course-data-content {{
      padding: 15px;
      display: none;
    }}
    .course-data-content.open {{
      display: block;
    }}
    .data-section {{
      margin-bottom: 15px;
      padding: 10px;
      background-color: white;
      border-left: 3px solid #007bff;
    }}
    .data-section h4 {{
      margin-top: 0;
      color: #333;
    }}

    /* ========== 馬レポートセクション ========== */
    .horse-report-card {{
      margin: 10px 0;
      border: 1px solid #ddd;
      border-radius: 5px;
      background-color: #fff;
    }}
    .horse-report-toggle {{
      cursor: pointer;
      user-select: none;
      padding: 12px;
      background-color: #e3f2fd;
      border: none;
      border-bottom: 1px solid #90caf9;
      border-radius: 5px 5px 0 0;
      font-weight: bold;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .horse-report-toggle:hover {{
      background-color: #bbdefb;
    }}
    .horse-report-toggle-icon {{
      font-size: 18px;
      transition: transform 0.3s;
    }}
    .horse-report-toggle-icon.open {{
      transform: rotate(180deg);
    }}
    .horse-report-content {{
      display: none;
      padding: 10px;
      background-color: #f9f9f9;
      border-radius: 0 0 5px 5px;
    }}
    .horse-report-content.open {{
      display: block;
    }}
  </style>
</head>
<body>
  <a id="pageTop"></a>
  {site_nav}
  {breadcrumb}
  <h2 class="race-page-title">{date_display}　{place_name}競馬場　第{race_num}R　{race_name}{course_link_html}</h2>
  <p class="race-page-time">発走時刻: {race_time_display}</p>
  {weather_info_html}
  {odds_update_label}
  <div class="table-wrap table-wrap--full">
  <table id="raceTable">
    <thead>
      <tr>
        <th>枠</th>
        <th>馬番 ▼</th>
        <th>馬名</th>
        <th>性齢</th>
        <th>斤量</th>
        <th>騎手</th>
        <th>馬体重</th>
        <th>人気 ▼</th>
        <th>単勝オッズ</th>
        <th class="mm-hitrate-col">的中率 ▼</th>
        <th class="mm-value-col">回収率 ▼</th>
        <th class="mm-mar-col">MAR ▼</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
  </div>
  {pick_summary_html}

  {ad_unit_1}

  {result_table_html}
  {payout_table_html}
  {book_html}

  {ad_unit_2}

  <!-- ========== コース別データセクション（折りたたみ可能） ========== -->
  <div class="course-data-section">
    <div class="course-data-header" onclick="toggleCourseData()">
      <span>コース別データ</span>
      <span class="course-data-toggle" id="courseDataToggle">▼</span>
    </div>
    <div class="course-data-content" id="courseDataContent">
      <div class="data-section">
        {run_time_info}
      </div>
      <div class="data-section">
        {weight_info}
      </div>
      <div class="data-section">
        {peds_info}
      </div>
      <div class="data-section">
        {pops_info}
      </div>
      <div class="data-section">
        {frames_info}
      </div>
      <div class="data-section">
        {recent_html}
      </div>
    </div>
  </div>

  {ad_unit_3}

  <h2 id="horseReportsSection">出走馬 詳細レポート</h2>
  <div id="horseReportsContainer"></div>

  <script>
  function toggleCourseData() {{
    const content = document.getElementById("courseDataContent");
    const toggle = document.getElementById("courseDataToggle");

    content.classList.toggle("open");
    toggle.classList.toggle("open");
  }}

  function toggleHorseReport(reportId) {{
    const content = document.getElementById(reportId);
    const toggle = content.previousElementSibling.querySelector('.horse-report-toggle-icon');

    content.classList.toggle('open');
    toggle.classList.toggle('open');
  }}

  function scrollToReport(reportId) {{
    // 詳細レポートセクションまでスクロール
    const element = document.getElementById(reportId);
    if (element) {{
      element.classList.add('open');
      element.previousElementSibling.querySelector('.horse-report-toggle-icon').classList.add('open');
      element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}
  }}

  document.addEventListener("DOMContentLoaded", () => {{
    // ======== スタイル設定部分 ========
    // 列順: 枠[0] 馬番[1] 馬名[2] 性齢[3] 斤量[4] 騎手[5] 馬体重[6]
    //       人気[7] 単勝オッズ[8] 的中率[9] 回収率[10] MAR[11]
    const rows = document.querySelectorAll("#raceTable tbody tr");
    rows.forEach(row => {{
      const waku = parseInt(row.children[0].innerText);
      row.children[0].classList.add(`waku-${{waku}}`);
      row.children[1].classList.add(`waku-${{waku}}`);
    }});
    // ======== ソート機能部分 ========
    const table = document.getElementById("raceTable");
    const headers = table.querySelectorAll("th");

    function getCellValue(tr, idx) {{
      const td = tr.children[idx];
      // data-sort 属性があれば数値として優先使用（指数列の確実なソート用）
      if (td && td.dataset.sort !== undefined && td.dataset.sort !== "") {{
        return Number(td.dataset.sort);
      }}
      const val = td ? td.innerText.trim() : "";
      return isNaN(val) || val === "" ? val : Number(val);
    }}

    function clearSortIndicators() {{
      headers.forEach(th => {{
        const ind = th.querySelector(".sort-ind");
        if (ind) ind.textContent = "";
      }});
    }}

    function sortTable(colIndex, th) {{
      const tbody = table.tBodies[0];
      const rowsArray = Array.from(tbody.querySelectorAll("tr"));

      // 前回の状態を取得（デフォルト asc）
      const currentDir = th.dataset.sortDir === "asc" ? "desc" : "asc";
      th.dataset.sortDir = currentDir;

      // 他ヘッダの矢印をリセット
      headers.forEach(header => {{
        if (header !== th) header.dataset.sortDir = "";
      }});

      // ソート方向アイコン
      clearSortIndicators();
      let indicator = th.querySelector(".sort-ind");
      if (!indicator) {{
        indicator = document.createElement("span");
        indicator.classList.add("sort-ind");
        indicator.style.marginLeft = "6px";
        th.appendChild(indicator);
      }}

      // ソート処理
      rowsArray.sort((a, b) => {{
        const A = getCellValue(a, colIndex);
        const B = getCellValue(b, colIndex);
        if (typeof A === "number" && typeof B === "number") {{
          return currentDir === "asc" ? A - B : B - A;
        }} else {{
          return currentDir === "asc"
            ? A.toString().localeCompare(B)
            : B.toString().localeCompare(A);
        }}
      }});

      // 並び替え反映
      rowsArray.forEach(r => tbody.appendChild(r));
    }}

    // ======== 対象列にクリックイベントを追加 ========
    // 馬番[1]、人気[7]、的中率[9]、回収率[10]、MAR[11] をソート可能に
    [1, 7, 9, 10, 11].forEach(idx => {{
      const th = headers[idx];
      if (th) {{
        th.style.cursor = "pointer";
        const indicator = document.createElement("span");
        indicator.classList.add("sort-ind");
        indicator.style.marginLeft = "6px";
        th.appendChild(indicator);
        th.addEventListener("click", () => sortTable(idx, th));
      }}
    }});
  }});
  </script>
  <p class="back-to-top"><a href="#pageTop">&uarr; ページの先頭へ戻る</a></p>
  {footer}
</body>
</html>
""".format(
    date_display=date_display,
    place_name=place_name,
    race_num=race_num,
    race_name=race_name,
    race_time_display=race_time_display,
    site_nav=site_nav,
    breadcrumb=breadcrumb,
    footer=footer,
    adsense=adsense,
    ga4=ga4,
    meta_tags=meta_tags,
    noindex_tag=noindex_tag,
    ad_unit_1=ad_unit_1,
    ad_unit_2=ad_unit_2,
    ad_unit_3=ad_unit_3,
    pick_summary_html=pick_summary_html,
    odds_update_label=odds_update_label,
    course_link_html=course_link_html,
    weather_info_html=weather_info_html,
    table_rows=table_rows,
    run_time_info=run_time_info,
    weight_info=weight_info,
    peds_info=peds_info,
    pops_info=pops_info,
    frames_info=frames_info,
    recent_html=recent_html,
    result_table_html=result_table_html,
    payout_table_html=payout_table_html,
    book_html=book_html,
)


def generate_result_table(df):
    if df.empty:
        return "<p>レース結果データが見つかりません。</p>"

    result_rows = ""
    for _, row in df.iterrows():
        rank = row["着順"]
        waku = row.get("枠", row.get("枠番", None))
        waku = waku if pd.notna(waku) else ""
        umaban = row["馬番"]
        horse = html.escape(str(row["馬名"]))
        jockey = html.escape(str(row["騎手"]))
        horse_weight = row["馬体重"] if "馬体重" in row and pd.notna(row["馬体重"]) else ""
        # 除外・取消等の馬はタイム・オッズがNaNになる（そのままf-stringに渡すと
        # 文字どおり"nan"が表示されてしまうため、未確定/対象外は空欄にする）
        time = row["タイム"] if pd.notna(row["タイム"]) else ""

        diff = row["着差"] if pd.notna(row["着差"]) else ""
        pop = str(int(float(row["人気"]))) if pd.notna(row["人気"]) else ""
        last_3f = row["上り"] if "上り" in row and pd.notna(row["上り"]) else ""
        race_position = row["通過"] if "通過" in row and pd.notna(row["通過"]) else ""
        odds = row["単勝"] if pd.notna(row["単勝"]) else ""
        # build_table_race_cardsと同じ理由でScore/Rankは常にバランス型（score/rank列）
        # を表示する（別の計算結果を混ぜるとScoreとRankの不整合が起きるため）
        score = row.get("score", "")
        pred_rank = row.get("rank", "")

        # --- 枠順背景色 ---
        waku_color = WAKU_COLORS.get(waku, "#ffffff")
        waku_style = f'background-color:{waku_color}; color:{"#fff" if waku in ["2","3","4","7"] else "#000"};'

        # --- 人気上位3頭色付け ---
        pop_color = RANK_COLORS.get(pop, "#ffffff")
        pop_style = f'background-color:{pop_color};'

        # --- Rank上位3頭色付け（人気と同系色だが薄いパステル系） ---
        pred_rank_color = PRED_RANK_COLORS.get(str(pred_rank), "")
        pred_rank_style = f'background-color:{pred_rank_color};' if pred_rank_color else ""

        # --- AI指数（score列は廃止、AI指数に統一）---
        ai_index = score_to_index(score) if isinstance(score, (int, float)) and pd.notna(score) else None

        # --- 馬体重の増減が大きい馬を強調 ---
        weight_style = _weight_change_style(horse_weight)

        result_rows += f"""
        <tr>
            <td>{rank}</td>
            <td style="{waku_style}">{waku}</td>
            <td style="{waku_style}">{umaban}</td>
            <td>{horse}</td>
            <td>{jockey}</td>
            <td style="{weight_style}">{horse_weight}</td>
            <td>{time}</td>
            <td>{diff}</td>
            <td style="{pop_style}">{pop}</td>
            <td>{last_3f}</td>
            <td>{race_position}</td>
            <td>{odds}</td>
            {_ai_index_cell_html(ai_index)}
            <td class="ai-rank-col" style="{pred_rank_style}">{pred_rank}</td>
        </tr>
        """

    result_table = f"""
    <h2>レース結果</h2>
    <div class="table-wrap table-wrap--full">
    <table id="resultTable">
      <thead>
        <tr>
          <th>着順</th><th>枠</th><th>馬番</th><th>馬名</th>
          <th>騎手</th><th>馬体重</th><th>タイム</th><th>着差</th>
          <th>人気</th><th>上り</th><th>通過</th>
          <th>単勝オッズ</th><th class="ai-score-col">AI指数</th><th class="ai-rank-col">Rank</th>
        </tr>
      </thead>
      <tbody>
        {result_rows}
      </tbody>
    </table>
    </div>
    """
    return result_table


def generate_payout_table_html(df):
    """指定されたレースIDに対応する配当結果テーブルをHTML化して返す"""
    if df.empty:
        return "<p>配当結果データが見つかりません。</p>"

    # --- 配当金額を3桁区切りに整形 ---
    df["配当"] = df["配当"].apply(lambda x: f"{int(x):,}円")

    # --- 同じ式別でまとめる ---
    grouped = (
        df.groupby("式別", sort=False)
        .apply(
            lambda g: pd.Series({
                "馬番": "<br>".join(g["馬番"].astype(str)),
                "配当": "<br>".join(g["配当"].astype(str)),
                "人気": "<br>".join(g["人気"].astype(int).astype(str))
            })
        )
        .reset_index()
    )

    # --- HTML構築 ---
    rows_html = ""
    for _, row in grouped.iterrows():
        rows_html += f"""
        <tr>
          <td>{row['式別']}</td>
          <td>{row['馬番']}</td>
          <td class="num">{row['配当']}</td>
          <td>{row['人気']}</td>
        </tr>
        """
    payout_html = f"""
    <h2>配当結果</h2>
    <div class="table-wrap table-wrap--full">
    <table id="payoutTable">
      <thead>
        <tr>
          <th>式別</th>
          <th>馬番</th>
          <th>配当</th>
          <th>人気</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    </div>
    """

    return payout_html


def _get_race_info_dict(target_id):
    """レース情報CSVから構造化データをdictで返す内部ヘルパー"""
    df_info = race_card_dataset_manager.get_race_info_csv(target_id)
    if df_info.empty:
        return None
    row = df_info.iloc[0]
    race_type = str(row.get("race_type", "") or "")
    course_len_raw = row.get("course_len", "")
    course_len = None
    if pd.notna(course_len_raw):
        try:
            course_len = int(float(str(course_len_raw).strip()))
        except Exception:
            pass
    weather = str(row.get("weather", "") or "")
    ground_state = str(row.get("ground_state", "") or "")
    race_class = str(row.get("class", "") or "")
    race_class = race_class.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    return dict(race_type=race_type, course_len=course_len, weather=weather,
                ground_state=ground_state, race_class=race_class)


def generate_race_info(date_str, place_id, target_id):
    """レース情報をcsvファイルから取得する（後方互換: 文字列を返す）"""
    info = _get_race_info_dict(target_id)
    if info is None:
        print(f"ℹ️ [スキップ/エラーではありません] レース情報未蓄積: {target_id}")
        return None
    cl = str(info["course_len"]) if info["course_len"] is not None else ""
    return (f"{info['race_type']}{cl}m "
            f"天候:{info['weather']} 馬場:{info['ground_state']} クラス:{info['race_class']}")


def generate_run_time_info(date_str, place_id, target_id):
    """平均勝ち時計/先週の三着内時計/ 同コース/条件 上りタイム"""
    # --- レース情報の取得 ---
    year = date_str[:4]
    race_type, course_len, ground_state, race_class = get_race_info(year, place_id, target_id)
    if race_type == None and course_len == None and ground_state == None and race_class == None:
        return

    # --- データ読み込み ---
    total_run_df = race_info_dataset_manager.get_total_average_time_csv(place_id)
    total_data_df = race_info_dataset_manager.get_total_winner_time_csv(place_id)
    year_run_df = race_info_dataset_manager.get_annual_average_time_csv(place_id, year)
    year_data_df = race_info_dataset_manager.get_annual_winner_time_csv(place_id, year)

    def normalize_ground_state(state):
        if pd.isna(state):
            return ""
        s = str(state)
        if "不" in s:
            return "不良"
        elif "稍" in s:
            return "稍重"
        elif "良" in s:
            return "良"
        elif "重" in s:
            return "重"
        return s

    # --- 該当行取得関数 ---
    def get_row(df, cls):
        if df.empty:
            return None
        gs = normalize_ground_state(ground_state)
        # ground_state 候補を柔軟に設定
        if gs == "不良":
            gs_candidates = ["不", "不良"]
        elif gs == "稍重":
            gs_candidates = ["稍", "稍重"]
        else:
            gs_candidates = [gs]
        cond = (
            (df["race_type"] == race_type) &
            (df["course_len"].astype(str) == str(course_len)) &
            (df["ground_state"].astype(str).apply(lambda x: any(cand in x for cand in gs_candidates))) &
            (df["class"] == cls)
        )
        sub = df[cond]
        if sub.empty:
            return None
        return sub.iloc[0]

    # --- 各行取得 ---
    # 勝ち時計: allクラス用
    year_all_time = get_row(year_run_df, "all")
    year_class_time = get_row(year_run_df, race_class)
    total_all_time = get_row(total_run_df, "all")
    total_class_time = get_row(total_run_df, race_class)

    # 上り/通過: 各クラス用
    year_all_data = get_row(year_data_df, "all")
    year_class_data = get_row(year_data_df, race_class)
    total_all_data = get_row(total_data_df, "all")
    total_class_data = get_row(total_data_df, race_class)

    # --- HTML整形用ユーティリティ ---
    def fmt_avg_time_html(row):
        """勝ち時計(ms) → mm:ss.ms 形式"""
        if row is None or pd.isna(row.get("avg_time", None)) or row["avg_time"] == "":
            return "―"
        try:
            val = float(row["avg_time"])
            total_seconds = int(val // 1000)
            ms = int(val % 1000)
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}:{seconds:02d}.{ms:03d}"
        except:
            return str(row["avg_time"])

    def color_for_position(pos):
        """通過順位の色を決定"""
        try:
            pos = int(pos)
        except:
            return "black"
        if 1 <= pos <= 2:
            return "red"
        elif 3 <= pos <= 9:
            return "orange"
        elif 10 <= pos <= 16:
            return "deepskyblue"
        elif pos >= 17:
            return "blue"
        return "black"

    def fmt_passing_html(row):
        """通過列（複数）を整形"""
        if row is None:
            return "―"
        passes = [row[col] for col in row.index if col.startswith("通過") and pd.notna(row[col]) and row[col] != ""]
        if not passes:
            return "―"
        html_parts = []
        for p in passes:
            color = color_for_position(p)
            html_parts.append(f'<span style="color:{color}; font-weight:bold;">{p}</span>')
        return "-".join(html_parts)

    def fmt_last_html(row):
        """上りタイムを整形"""
        if row is None or "上り" not in row or pd.isna(row["上り"]) or row["上り"] == "":
            return "―"
        return f"{row['上り']}"

    # --- HTML整形 ---
    run_time_info_html = f"""
    <div id="runtimeInfo" style="margin: 20px 0; padding: 10px; border: 1px solid #ccc; background: #fafafa;">
      <h3>🕐 コース別平均タイム情報 ({race_type} {course_len}m {ground_state} {race_class})</h3>
      <div class="table-wrap">
      <table style="border-collapse: collapse; text-align: center;">
        <thead>
          <tr style="background: #f2f2f2;">
            <th>区分</th>
            <th>対象</th>
            <th>平均勝ち時計</th>
            <th>上り</th>
            <th>通過</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td rowspan="2">全クラス</td>
            <td>{year}年平均</td>
            <td>{fmt_avg_time_html(year_all_time)}</td>
            <td>{fmt_last_html(year_all_data)}</td>
            <td>{fmt_passing_html(year_all_data)}</td>
          </tr>
          <tr>
            <td>TOTAL平均</td>
            <td>{fmt_avg_time_html(total_all_time)}</td>
            <td>{fmt_last_html(total_all_data)}</td>
            <td>{fmt_passing_html(total_all_data)}</td>
          </tr>
          <tr>
            <td rowspan="2">{race_class}</td>
            <td>{year}年平均</td>
            <td>{fmt_avg_time_html(year_class_time)}</td>
            <td>{fmt_last_html(year_class_data)}</td>
            <td>{fmt_passing_html(year_class_data)}</td>
          </tr>
          <tr>
            <td>TOTAL平均</td>
            <td>{fmt_avg_time_html(total_class_time)}</td>
            <td>{fmt_last_html(total_class_data)}</td>
            <td>{fmt_passing_html(total_class_data)}</td>
          </tr>
        </tbody>
      </table>
      </div>
    </div>
    """.strip()

    return run_time_info_html


def generate_weight_info(date_str, place_id, target_id):
    """勝ち馬の平均馬体重を取得して HTML を生成する。"""
    # --- レース情報の取得 ---
    year = date_str[:4]
    race_type, course_len, ground_state, race_class = get_race_info(year, place_id, target_id)
    if race_type == None and course_len == None and ground_state == None and race_class == None:
        return

    total_df = race_info_dataset_manager.get_total_winner_weight_csv(place_id)
    year_df = race_info_dataset_manager.get_annual_winner_weight_csv(place_id, year)

    def get_row(df, cls):
        if df.empty:
            return None
        cond = (
            (df["race_type"] == race_type) &
            (df["course_len"].astype(str) == str(course_len)) &
            (df["ground_state"] == ground_state) &
            (df["class"] == cls)
        )
        sub = df[cond]
        if sub.empty:
            return None
        return sub.iloc[0]

    year_all = get_row(year_df, "all")
    year_class = get_row(year_df, race_class)
    total_all = get_row(total_df, "all")
    total_class = get_row(total_df, race_class)

    def fmt_weight_html(row):
        """馬体重に色をつけて表示"""
        if row is None or "馬体重" not in row or pd.isna(row["馬体重"]) or row["馬体重"] == "":
            return "―"
        try:
            weight = float(row["馬体重"])
            if weight >= 500:
                color = "red"
            elif weight <= 450:
                color = "deepskyblue"
            else:
                color = "black"
            return f'<span style="color:{color}; font-weight:bold;">{weight:.1f}kg</span>'
        except:
            return str(row["馬体重"])

    # --- HTML生成 ---
    weight_info_html = f"""
    <div id="weightInfo" style="margin: 20px 0; padding: 10px; border: 1px solid #ccc; background: #fefefe;">
      <h3>🐎 コース別平均馬体重情報 ({race_type} {course_len}m {ground_state} {race_class})</h3>
      <div class="table-wrap">
      <table style="border-collapse: collapse; text-align: center;">
        <thead>
          <tr style="background: #f2f2f2;">
            <th>区分</th>
            <th>対象</th>
            <th>平均馬体重</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td rowspan="2">全クラス</td>
            <td>{year}年平均</td>
            <td>{fmt_weight_html(year_all)}</td>
          </tr>
          <tr>
            <td>TOTAL平均</td>
            <td>{fmt_weight_html(total_all)}</td>
          </tr>
          <tr>
            <td rowspan="2">{race_class}</td>
            <td>{year}年平均</td>
            <td>{fmt_weight_html(year_class)}</td>
          </tr>
          <tr>
            <td>TOTAL平均</td>
            <td>{fmt_weight_html(total_class)}</td>
          </tr>
        </tbody>
      </table>
      </div>
    </div>
    """.strip()

    return weight_info_html


def generate_peds_result_html(date_str, place_id, target_id):
    """血統別成績（PedsResults）をHTMLで整形して返す"""
    # --- レース情報の取得 ---
    year = date_str[:4]
    race_type, course_len, ground_state, race_class = get_race_info(year, place_id, target_id)
    if race_type == None and course_len == None and ground_state == None and race_class == None:
        return

    total_df = peds_results_dataset_manager.get_total_peds_results_csv(place_id, race_type, course_len, ground_state)
    year_df = peds_results_dataset_manager.get_annual_peds_results_csv(place_id, year, race_type, course_len, ground_state)

    # --- どちらも空なら表示なし ---
    if total_df.empty and year_df.empty:
        return f"""
        <div class="peds-result-block"; style="margin: 20px 0; padding: 10px; border: 1px solid #ccc; background: #fefefe;">
          <h3>🐎 コース別血統成績 ({race_type} {course_len}m {ground_state})</h3>
          <p style="color:#888;">データが存在しません。</p>
        </div>
        """

    # --- クラス順保持 ---
    CLASS_ORDER = ["all", "未勝利", "新馬", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"]
    # --- クラス列が存在する場合のみカテゴリ化 ---
    for df in [total_df, year_df]:
        if not df.empty and "クラス" in df.columns:
            df["クラス"] = pd.Categorical(df["クラス"], categories=CLASS_ORDER, ordered=True)

    def make_table_html(df, cls_name, title):
        """サブテーブル作成"""
        if df.empty or "クラス" not in df.columns:
            return f"<h4>{title}：データなし</h4>"

        sub = df[df["クラス"] == cls_name].copy()
        if sub.empty:
            return f"<h4>{title}：データなし</h4>"

        # 上位5件（1着多い順→2着→3着）
        sub = sub.sort_values(by=["1着", "2着", "3着"], ascending=False).head(5)
        sub["総数"] = sub[["1着", "2着", "3着", "着外"]].sum(axis=1)
        # 総数が0（対象レースなし）の場合は0/0でNaNになり、そのままだと"nan%"と
        # 表示されてしまうため、対象なしは0%にする
        sub["勝率"] = (sub["1着"] / sub["総数"] * 100).round(1).fillna(0)
        sub["複勝率"] = ((sub["1着"] + sub["2着"] + sub["3着"]) / sub["総数"] * 100).round(1).fillna(0)

        rows = ""
        for _, r in sub.iterrows():
            stat = f"({int(r['1着'])},{int(r['2着'])},{int(r['3着'])},{int(r['着外'])})"
            rows += f"""
            <tr>
              <td>{r['血統']}</td>
              <td>{stat}</td>
              <td>{r['勝率']}%</td>
              <td>{r['複勝率']}%</td>
            </tr>"""

        html = f"""
        <h4>{title} </h4>
        <div class="table-wrap">
        <table class="peds-table">
          <thead>
            <tr><th>血統</th><th>成績(1,2,3,着外)</th><th>勝率</th><th>複勝率</th></tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        </div>
        """
        return html

    # --- 出力HTML作成 ---
    html = f"""
    <div class="peds-result-block"; style="margin: 20px 0; padding: 10px; border: 1px solid #ccc; background: #fefefe;">
      <h3>🧬 血統別成績 ({race_type} {course_len}m {ground_state})</h3>

      {make_table_html(total_df, "all", f"全クラス 2019~{year}")}
      {make_table_html(total_df, race_class, f"{race_class} 2019~{year}")}
      {make_table_html(year_df, "all", f"全クラス {year}年")}
      {make_table_html(year_df, race_class, f"{race_class} {year}年")}
    </div>
    """

    return html


def generate_pops_info(date_str, place_id, target_id):
    """勝ち馬の平均人気と3着内平均人気を取得して HTML を生成する。"""
    # --- レース情報の取得 ---
    year = date_str[:4]
    race_type, course_len, ground_state, race_class = get_race_info(year, place_id, target_id)
    if race_type == None and course_len == None and ground_state == None and race_class == None:
        return

    def normalize_class_column(df):
        if not df.empty and "class" in df.columns:
            trans_table = str.maketrans("０１２３４５６７８９", "0123456789")
            df["class"] = df["class"].astype(str).apply(lambda x: x.translate(trans_table).strip())
        return df

    total_df = normalize_class_column(race_info_dataset_manager.get_total_average_pops_csv(place_id, top3=False))
    year_df = normalize_class_column(race_info_dataset_manager.get_annual_average_pops_csv(place_id, year, top3=False))
    total_top3_df = normalize_class_column(race_info_dataset_manager.get_total_average_pops_csv(place_id, top3=True))
    year_top3_df = normalize_class_column(race_info_dataset_manager.get_annual_average_pops_csv(place_id, year, top3=True))

    def get_row(df, cls):
        if df.empty:
            return None
        # クラス名を全角→半角に変換しておく
        trans_table = str.maketrans("０１２３４５６７８９", "0123456789")
        cls = str(cls).translate(trans_table).strip()
        cond = (
            (df["race_type"] == race_type) &
            (df["course_len"].astype(str) == str(course_len)) &
            (df["ground_state"] == ground_state) &
            (df["class"] == cls)
        )
        sub = df[cond]
        if sub.empty:
            return None
        return sub.iloc[0]

    year_all = get_row(year_df, "all")
    year_class = get_row(year_df, race_class)
    total_all = get_row(total_df, "all")
    total_class = get_row(total_df, race_class)

    year_all_top3 = get_row(year_top3_df, "all")
    year_class_top3 = get_row(year_top3_df, race_class)
    total_all_top3 = get_row(total_top3_df, "all")
    total_class_top3 = get_row(total_top3_df, race_class)

    # --- 表示フォーマット ---
    def fmt_pops_html(pops_value):
        """人気数値に色をつけて表示"""
        if pops_value is None or pops_value == "" or pd.isna(pops_value):
            return "―"
        try:
            pops = float(pops_value)
            if pops >= 12:
                color = "red"
            elif pops >= 6:
                color = "deepskyblue"
            else:
                color = "black"
            return f'<span style="color:{color}; font-weight:bold;">{pops:.1f}番人気</span>'
        except:
            return str(pops_value)

    # --- HTML生成 ---
    pops_info_html = f"""
    <div id="popsInfo" style="margin: 20px 0; padding: 10px; border: 1px solid #ccc; background: #fefefe;">
      <h3>📊 コース別平均人気情報 ({race_type} {course_len}m {ground_state} {race_class})</h3>
      <div class="table-wrap">
      <table style="border-collapse: collapse; text-align: center;">
        <thead>
          <tr style="background: #f2f2f2;">
            <th>区分</th>
            <th>対象</th>
            <th>平均勝馬人気</th>
            <th>平均着内人気</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td rowspan="2">全クラス</td>
            <td>{year}年平均</td>
            <td>{fmt_pops_html(year_all["avg_pop"]) if year_all is not None else "―"}</td>
            <td>{fmt_pops_html(year_all_top3["avg_pop"]) if year_all_top3 is not None else "―"}</td>
          </tr>
          <tr>
            <td>TOTAL平均</td>
            <td>{fmt_pops_html(total_all["avg_pop"]) if total_all is not None else "―"}</td>
            <td>{fmt_pops_html(total_all_top3["avg_pop"]) if total_all_top3 is not None else "―"}</td>
          </tr>
          <tr>
            <td rowspan="2">{race_class}</td>
            <td>{year}年平均</td>
            <td>{fmt_pops_html(year_class["avg_pop"]) if year_class is not None else "―"}</td>
            <td>{fmt_pops_html(year_class_top3["avg_pop"]) if year_class_top3 is not None else "―"}</td>
          </tr>
          <tr>
            <td>TOTAL平均</td>
            <td>{fmt_pops_html(total_class["avg_pop"]) if total_class is not None else "―"}</td>
            <td>{fmt_pops_html(total_class_top3["avg_pop"]) if total_class_top3 is not None else "―"}</td>
          </tr>
        </tbody>
      </table>
      </div>
    </div>
    """.strip()

    return pops_info_html


def generate_frame_horse_info(date_str, place_id, target_id):
    """勝ち馬と3着内馬の平均枠番・平均馬番を取得して HTML を生成する。"""
    # --- レース情報の取得 ---
    year = date_str[:4]
    race_type, course_len, ground_state, race_class = get_race_info(year, place_id, target_id)
    if race_type is None:
        return ""

    def normalize_class_column(df):
        if not df.empty and "class" in df.columns:
            trans_table = str.maketrans("０１２３４５６７８９", "0123456789")
            df["class"] = df["class"].astype(str).apply(lambda x: x.translate(trans_table).strip())
        return df

    total_df = normalize_class_column(race_info_dataset_manager.get_total_average_frames_csv(place_id, top3=False))
    total_top3_df = normalize_class_column(race_info_dataset_manager.get_total_average_frames_csv(place_id, top3=True))
    year_df = normalize_class_column(race_info_dataset_manager.get_annual_average_frames_csv(place_id, year, top3=False))
    year_top3_df = normalize_class_column(race_info_dataset_manager.get_annual_average_frames_csv(place_id, year, top3=True))

    def get_row(df, cls):
        if df.empty:
            return None
        # クラス名の統一（全角→半角）
        trans_table = str.maketrans("０１２３４５６７８９", "0123456789")
        cls = str(cls).translate(trans_table).strip()
        cond = (
            (df["race_type"] == race_type)
            & (df["course_len"].astype(str) == str(course_len))
            & (df["ground_state"] == ground_state)
            & (df["class"] == cls)
        )
        sub = df[cond]
        if sub.empty:
            return None
        return sub.iloc[0]

    # --- 該当行取得 ---
    total_class = get_row(total_df, race_class)
    total_all = get_row(total_df, "all")
    total_top3_class = get_row(total_top3_df, race_class)
    total_top3_all = get_row(total_top3_df, "all")

    year_class = get_row(year_df, race_class)
    year_all = get_row(year_df, "all")
    year_top3_class = get_row(year_top3_df, race_class)
    year_top3_all = get_row(year_top3_df, "all")

    # --- HTML整形 ---
    def fmt_frame_color(value):
        """枠番の色付け：1〜2=赤、7〜8=青、それ以外=黒"""
        if value is None or value == "" or pd.isna(value):
            return "―"
        try:
            val = float(value)
            if val in [1, 2]:
                color = "red"
            elif val in [7, 8]:
                color = "deepskyblue"
            else:
                color = "black"
            return f'<span style="color:{color}; font-weight:bold;">{val:.2f}</span>'
        except:
            return str(value)

    def fmt_horse_color(value):
        """馬番の色付け：1〜4=赤、13〜18=青、それ以外=黒"""
        if value is None or value == "" or pd.isna(value):
            return "―"
        try:
            val = float(value)
            if 1 <= val <= 4:
                color = "red"
            elif 13 <= val <= 18:
                color = "deepskyblue"
            else:
                color = "black"
            return f'<span style="color:{color}; font-weight:bold;">{val:.2f}</span>'
        except:
            return str(value)

    # --- HTML生成 ---
    html = f"""
    <div id="frameHorseInfo" style="margin:20px 0; padding:10px; border:1px solid #ccc; background:#fefefe;">
      <h3>📊 枠番・馬番 平均情報 ({race_type} {course_len}m {ground_state} {race_class})</h3>
      <div class="table-wrap">
      <table style="border-collapse:collapse; text-align:center;">
        <thead>
          <tr style="background:#f2f2f2;">
            <th>区分</th>
            <th>対象</th>
            <th>平均枠番(勝ち馬)</th>
            <th>平均馬番(勝ち馬)</th>
            <th>平均枠番(3着内)</th>
            <th>平均馬番(3着内)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td rowspan="2">全クラス</td>
            <td>{year}年平均</td>
            <td>{fmt_frame_color(year_all["avg_frame"]) if year_all is not None else "―"}</td>
            <td>{fmt_horse_color(year_all["avg_horse"]) if year_all is not None else "―"}</td>
            <td>{fmt_frame_color(year_top3_all["avg_frame"]) if year_top3_all is not None else "―"}</td>
            <td>{fmt_horse_color(year_top3_all["avg_horse"]) if year_top3_all is not None else "―"}</td>
          </tr>
          <tr>
            <td>TOTAL平均</td>
            <td>{fmt_frame_color(total_all["avg_frame"]) if total_all is not None else "―"}</td>
            <td>{fmt_horse_color(total_all["avg_horse"]) if total_all is not None else "―"}</td>
            <td>{fmt_frame_color(total_top3_all["avg_frame"]) if total_top3_all is not None else "―"}</td>
            <td>{fmt_horse_color(total_top3_all["avg_horse"]) if total_top3_all is not None else "―"}</td>
          </tr>
          <tr>
            <td rowspan="2">{race_class}</td>
            <td>{year}年平均</td>
            <td>{fmt_frame_color(year_class["avg_frame"]) if year_class is not None else "―"}</td>
            <td>{fmt_horse_color(year_class["avg_horse"]) if year_class is not None else "―"}</td>
            <td>{fmt_frame_color(year_top3_class["avg_frame"]) if year_top3_class is not None else "―"}</td>
            <td>{fmt_horse_color(year_top3_class["avg_horse"]) if year_top3_class is not None else "―"}</td>
          </tr>
          <tr>
            <td>TOTAL平均</td>
            <td>{fmt_frame_color(total_class["avg_frame"]) if total_class is not None else "―"}</td>
            <td>{fmt_horse_color(total_class["avg_horse"]) if total_class is not None else "―"}</td>
            <td>{fmt_frame_color(total_top3_class["avg_frame"]) if total_top3_class is not None else "―"}</td>
            <td>{fmt_horse_color(total_top3_class["avg_horse"]) if total_top3_class is not None else "―"}</td>
          </tr>
        </tbody>
      </table>
      </div>
    </div>
    """.strip()

    return html


def generate_recent_same_condition_html(date_str, place_id, target_id):
    """近10日間の同条件レース上位3頭をHTMLで表示する"""
    # --- 基準レース情報取得 ---
    year = date_str[:4]
    base_type, base_len, ground_state, race_class = get_race_info(year, place_id, target_id)
    if base_type == None and base_len == None and ground_state == None and race_class == None:
        return

    # --- 日付処理 ---
    base_date = datetime.strptime(date_str, "%Y%m%d")
    recent_days = sorted([base_date - timedelta(days=i) for i in range(1, 11)])
    matched_race_ids = []
    # --- 各日付ごとに処理 ---
    for race_day in recent_days:
        race_day_str = race_day.strftime("%Y%m%d")
        try:
            daily_ids = race_schedule_dataset_manager.get_daily_id(place_id, race_day)
        except Exception:
            continue

        for rid in daily_ids:
            info = get_race_info(year, place_id, rid)
            if info is None:
                continue
            race_type, course_len, ground_state, race_class = info

            # 条件一致
            if race_type == base_type and str(course_len) == str(base_len):
                matched_race_ids.append((rid, race_day_str, race_class, ground_state))

    if not matched_race_ids:
        return "<div>同条件の近走レースはありません。</div>"

    # --- HTML構築開始 ---
    html = f"""
    <div id="recentSameCondition" style="margin-top:20px; padding:10px; border:1px solid #ccc; background:#fefefe;">
      <h3>🏁 先週/今週の{NAME_LIST[place_id - 1]} {base_type} {base_len}m レース結果</h3>
    """

    race_day_dt = datetime.strptime(date_str, "%Y%m%d")
    df_info = race_card_dataset_manager.get_race_time_id_list_df(race_day_dt)

    for race_id, race_date_str, race_class, ground_state in matched_race_ids:
        df_all = race_result_dataset_manager.get_race_id_result(race_id)
        if df_all.empty:
            print(f"ℹ️ [スキップ/エラーではありません] レース結果データなし（未確定）: {race_id}")
            continue

        # 上位3頭抽出
        df_top3 = df_all.head(3)[["馬名", "タイム", "人気", "単勝", "上り", "通過", "馬体重"]]

        # レース情報
        type, len, ground, race_class_name = get_race_info(year, place_id, race_id)
        race_num = str(int(race_id[-2:]))

        # レース名
        race_name = ""
        if not df_info.empty:
            match = df_info[df_info["race_id"].astype(str) == str(race_id)]
            if not match.empty:
                race_name = str(match.iloc[0]["race_name"])
        race_date_dsp = datetime.strptime(race_date_str, "%Y%m%d").strftime("%Y/%m/%d")
        # --- HTML組み立て ---
        html += f"""
        <div style="margin-top:10px; padding:5px; border:1px solid #ddd;">
          <h4>{race_date_dsp}:{race_num}R {race_name} {type}{len}m {race_class_name} ({ground})</h4>
          <div class="table-wrap">
          <table style="border-collapse:collapse; text-align:center; font-size:14px;">
            <thead>
              <tr style="background:#f2f2f2;">
                <th>順位</th><th>馬名</th><th>タイム</th><th>人気</th><th>単勝</th><th>上り</th><th>通過</th><th>馬体重</th>
              </tr>
            </thead>
            <tbody>
        """

        for i, row in df_top3.iterrows():
            # 順位の色付け
            result_rank = int(i + 1)
            result_rank_text_color = RANK_COLORS.get(str(result_rank), "#ffffff")
            result_rank_html = f'<td style="background-color: {result_rank_text_color}; font-weight: bold;">{result_rank}</td>'
            # 人気の色付け
            popularity = str(row["人気"]).strip()
            # 2桁人気は赤色のテキスト
            try:
                popularity = int(float(popularity))
                pop_color = RANK_COLORS.get(str(popularity), "#ffffff")
                pop_text_color = "red" if popularity >= 10 else "black"
            except:
                popularity = int(99)
                pop_color = "#ffffff"
                pop_text_color = "black"
            pop_html = f'<td style="background-color: {pop_color}; color: {pop_text_color}; font-weight: bold;">{popularity}</td>'

            odds_html = _odds_cell_html(row["単勝"])
            # 時間表記を修正
            time_raw = row["タイム"]
            try:
                time_raw = re.sub(r"^0:", "", time_raw)
            except Exception:
                time_raw = time_raw

            html += f"""
            <tr>
              {result_rank_html}
              <td>{row["馬名"]}</td>
              <td>{time_raw}</td>
              {pop_html}
              {odds_html}
              <td>{row["上り"]}</td>
              <td>{row["通過"]}</td>
              <td>{row["馬体重"]}</td>
            </tr>
          """

        html += "</tbody></table></div></div>"

    html += "</div>"
    return html


def make_race_card_html(date_str, place_id, target_id):
    """レースカード HTML を生成して保存する"""
    race_num = int(str(target_id)[-2:])
    filename = f"{PLACE_LIST[place_id - 1]}R{race_num}.html"
    date_display = format_date(date_str)

    # CSV読込
    df = read_race_csv(date_str, target_id)
    if df is None:
        return

    # --- レース情報（コース・距離・馬場・クラス）を取得 ---
    race_info_dict = _get_race_info_dict(target_id)

    # --- レース結果、配当取得 ---
    result_df = get_result_table(date_str, place_id, target_id)
    if not result_df.empty:
        result_df = merge_rank_score(result_df, df)
    result_table_html = generate_result_table(result_df)

    returns_df = get_returns_table(date_str, place_id, target_id)
    payout_table_html = generate_payout_table_html(returns_df)

    is_confirmed = not result_df.empty

    # 確定結果がある場合、race_card の人気・オッズが未取得なら result_df から補完する
    if is_confirmed and "馬番" in df.columns:
        if "人気" in result_df.columns:
            pop_map = result_df.set_index(result_df["馬番"].astype(str))["人気"].to_dict()
            current_pop = df["人気"] if "人気" in df.columns else pd.Series([""] * len(df), index=df.index)
            needs_pop = current_pop.apply(lambda v: str(v) not in [str(i) for i in range(1, 19)])
            if needs_pop.any():
                df = df.copy()
                if "人気" not in df.columns:
                    df["人気"] = ""
                df.loc[needs_pop, "人気"] = df.loc[needs_pop, "馬番"].astype(str).map(pop_map)
        if "単勝" in result_df.columns:
            odds_map = result_df.set_index(result_df["馬番"].astype(str))["単勝"].to_dict()
            current_odds = df["オッズ"] if "オッズ" in df.columns else pd.Series([""] * len(df), index=df.index)
            needs_odds = current_odds.apply(lambda v: str(v).strip() in ["", "---.-", "nan"])
            if needs_odds.any():
                df = df.copy()
                if "オッズ" not in df.columns:
                    df["オッズ"] = ""
                df.loc[needs_odds, "オッズ"] = df.loc[needs_odds, "馬番"].astype(str).map(odds_map)

    # テーブル行作成
    table_rows = build_table_race_cards(df)

    # レース名・時刻取得
    race_day = datetime.strptime(date_str, "%Y%m%d")
    df_info = race_card_dataset_manager.get_race_time_id_list_df(race_day)
    race_name = ""
    race_time = ""
    if not df_info.empty:
        match = df_info[df_info["race_id"].astype(str) == str(target_id)]
        if not match.empty:
            race_name = str(match.iloc[0]["race_name"])
            race_time = str(match.iloc[0]["race_time"])

    # レースの平均時計、上り時計を取得
    run_time_info = generate_run_time_info(date_str, place_id, target_id)
    # レースの勝ち馬の平均馬体重情報を取得
    weight_info = generate_weight_info(date_str, place_id, target_id)
    # レースの血統情報を取得
    peds_info = generate_peds_result_html(date_str, place_id, target_id)
    # レースの人気情報を取得
    pops_info = generate_pops_info(date_str, place_id, target_id)
    # レースの枠順情報を取得
    frames_info = generate_frame_horse_info(date_str, place_id, target_id)
    # 近走の結果を取得
    recent_html = generate_recent_same_condition_html(date_str, place_id, target_id)

    # --- 各馬の詳細レポートを生成（折りたたみ可能に） ---
    horse_reports_html = """
    <style>
      .horse-report-toggle {
        cursor: pointer;
        user-select: none;
        padding: 12px;
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      .horse-report-toggle:hover {
        background-color: #bbdefb;
      }
      .horse-report-toggle-icon {
        font-size: 18px;
        transition: transform 0.3s;
      }
      .horse-report-toggle-icon.open {
        transform: rotate(180deg);
      }
      .horse-report-content {
        display: none;
        padding: 10px;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-top: none;
        border-radius: 0 0 5px 5px;
      }
      .horse-report-content.open {
        display: block;
      }
    </style>
    """

    for idx, (_, row) in enumerate(df.iterrows()):
        # 枠順未確定時は枠・馬番がNaNになり、str(nan)で文字どおり"nan"が表示されて
        # しまう（開発者目線の値でユーザーには不具合に見えるため、未確定は"-"にする）
        waku_raw = row.get("枠", "")
        umaban_raw = row.get("馬番", "")
        waku = str(int(float(waku_raw))) if pd.notna(waku_raw) and str(waku_raw).strip() != "" else "-"
        umaban = str(int(float(umaban_raw))) if pd.notna(umaban_raw) and str(umaban_raw).strip() != "" else "-"
        horse_name = str(row.get("馬名", "")).strip()
        if not horse_name:
            continue

        try:
            # 🐴 各馬の血統・成績・持ち時計を取得
            report = horse_report_generator.build_horse_report(
                horse_name,
                place_id,
                target_id,
                date_str
            )
            # AI指数・rank・人気を rowから取得して総評に渡す
            _score_raw = row.get("score", "")
            try:
                _ai_idx = score_to_index(float(_score_raw)) if _score_raw != "" and pd.notna(_score_raw) else None
            except Exception:
                _ai_idx = None
            try:
                _rank = int(row.get("rank", "")) if pd.notna(row.get("rank", "")) else None
            except Exception:
                _rank = None
            try:
                _pop = int(float(row.get("人気", ""))) if pd.notna(row.get("人気", "")) else None
            except Exception:
                _pop = None
            # 🧩 HTML化
            report_html = horse_report_generator.horse_report_to_html(report, _ai_idx, _rank, _pop)

            # 折りたたみセクションでラップ
            unique_id = f"horse_report_{idx}_{umaban}"
            horse_reports_html += f"""
            <div class="horse-report-card">
              <div class="horse-report-toggle" onclick="toggleHorseReport('{unique_id}')">
                <span>🐎 [{waku}枠{umaban}番] {horse_name}</span>
                <span class="horse-report-toggle-icon open">▼</span>
              </div>
              <div class="horse-report-content open" id="{unique_id}">
                {report_html}
              </div>
            </div>
            """
        except Exception as e:
            print(f"❌ [要確認/エラー] {horse_name} のレポート作成に失敗: {e}")
            continue

    # コース情報とヘッダー用HTMLを事前に組み立てる
    course_link_html = ""
    weather_info_html = ""
    if race_info_dict and race_info_dict["race_type"] and race_info_dict["course_len"]:
        rt = race_info_dict["race_type"]
        cl = race_info_dict["course_len"]
        place_key_c = PLACE_LIST[place_id - 1]
        course_href = f"../../courses/{place_key_c}/{rt}-{cl}.html"
        type_class = "turf" if rt in ("芝", "障害") else "dirt"
        course_link_html = f'<a href="{course_href}" class="course-link {type_class}">{rt}{cl}m</a>'
    if race_info_dict:
        weather_info_html = (
            f'<p class="race-page-info">'
            f'天候:{race_info_dict["weather"]} '
            f'馬場:{race_info_dict["ground_state"]} '
            f'クラス:{race_info_dict["race_class"]}'
            f'</p>'
        )

    # --- HTML生成・書き込み ---
    html_content = build_html_content(
        date_str=date_str,
        date_display=date_display,
        place_id=place_id,
        race_num=race_num,
        race_name=race_name,
        race_time=race_time,
        target_id=target_id,
        table_rows=table_rows,
        run_time_info=run_time_info,
        weight_info=weight_info,
        peds_info=peds_info,
        pops_info=pops_info,
        frames_info=frames_info,
        recent_html=recent_html,
        result_table_html=result_table_html,
        payout_table_html=payout_table_html,
        pick_summary_html=build_ai_pick_summary_html(df),
        odds_update_label=_odds_update_label_html(date_str, target_id, is_confirmed),
        course_link_html=course_link_html,
        weather_info_html=weather_info_html,
    )

    # 🧩 各馬の詳細レポートを race_page に追加（折りたたみ機能付き）
    html_content = html_content.replace(
        '<div id="horseReportsContainer"></div>',
        f'<div id="horseReportsContainer">{horse_reports_html}</div>'
    )

    html_manager.save_race_page_html(date_str, filename, html_content)


def make_daily_race_card_html(race_day=date.today()):
    """指定された日付の全レースカード HTML を生成する

    右側タブ・パンくず（_race_card_breadcrumb_items）は、生成済みページの有無では
    なく開催スケジュール（race_card_dataset_manager.get_race_time_id_list_df）から
    リンク先を決めるため、生成順に依存しない。1回のループで全レースぶん生成すれば
    すべて確定する。
    """
    date_str = race_day.strftime("%Y%m%d")

    targets = []
    for place_id in range(1, len(PLACE_LIST) + 1):
        race_id_list = race_schedule_dataset_manager.get_daily_id(place_id, race_day)
        if not race_id_list:
            print(f"ℹ️ [スキップ/エラーではありません] 指定日に開催なし: {date_str} {PLACE_LIST[place_id - 1]}")
            continue
        for race_id in race_id_list:
            targets.append((place_id, race_id))

    for place_id, race_id in targets:
        make_race_card_html(date_str, place_id, race_id)


def make_up_to_date_race_card_html(start_day=date(2025, 10, 1), today=date.today()):
    while start_day <= today:
        print(f"🏇 {start_day} のレースカードを作成中...")
        make_daily_race_card_html(start_day)
        start_day += timedelta(days=1)
    print("🎉 すべての日付のレースカード作成が完了しました！")
