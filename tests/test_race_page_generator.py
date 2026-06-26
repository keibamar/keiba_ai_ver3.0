import re
import shutil
from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.config.constants import NAME_LIST, PLACE_LIST
from src.logic.html_generator import race_page_generator as r
from src.managers import race_card_dataset_manager

SAMPLE_DATE_STR = "20241020"
SAMPLE_RACE_DAY = date(2024, 10, 20)
SAMPLE_PLACE_ID = 4
SAMPLE_RACE_ID = "202404040601"


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    """race_card/race_info/public_htmlの出力先をtmp_path配下に切り替える。

    average_pops/weights/frames/times・race_time_id_list等は実データ
    （04_nigata, 2024年）をそのまま参照する。
    """
    monkeypatch.setattr(paths, "RACE_CARD_DATA_PATH", str(tmp_path / "race_card"))
    monkeypatch.setattr(paths, "RACE_INFO_DATA_PATH", str(tmp_path / "race_info"))
    monkeypatch.setattr(paths, "PUBLIC_HTML_RACES_PATH", str(tmp_path / "public_html" / "races"))

    # 出走表+score/rank（実データをコピー）
    race_card_dir = tmp_path / "race_card" / SAMPLE_DATE_STR
    race_card_dir.mkdir(parents=True)
    shutil.copy(
        f"data/race_card/{SAMPLE_DATE_STR}/{SAMPLE_RACE_ID}.csv",
        race_card_dir / f"{SAMPLE_RACE_ID}.csv",
    )

    # レース情報（data/race_result/04_nigata/2024_race_results.csv から抽出した実際の値）
    race_info_df = pd.DataFrame({
        "race_type": ["芝"],
        "course_len": ["2000"],
        "weather": ["晴"],
        "ground_state": ["稍重"],
        "class": ["未勝利"],
    })
    race_card_dataset_manager.save_race_info_df(race_info_df, SAMPLE_RACE_DAY, SAMPLE_RACE_ID)

    return tmp_path


def test_get_race_info_returns_real_values(new_roots):
    race_type, course_len, ground_state, race_class = r.get_race_info("2024", SAMPLE_PLACE_ID, SAMPLE_RACE_ID)
    assert race_type == "芝"
    assert course_len == 2000
    assert ground_state == "稍重"
    assert race_class == "未勝利"


def test_generate_race_info_text(new_roots):
    text = r.generate_race_info(SAMPLE_DATE_STR, SAMPLE_PLACE_ID, SAMPLE_RACE_ID)
    assert text == "芝2000m 天候:晴 馬場:稍重 クラス:未勝利"


def test_generate_run_time_info_has_real_data(new_roots):
    html_str = r.generate_run_time_info(SAMPLE_DATE_STR, SAMPLE_PLACE_ID, SAMPLE_RACE_ID)
    assert 'id="runtimeInfo"' in html_str
    assert "コース別平均タイム情報" in html_str
    assert "平均勝ち時計" in html_str
    # TOTAL平均（2019〜）には実データが存在するため、mm:ss.ms形式の時計が表示される
    assert re.search(r"\d:\d{2}\.\d{3}", html_str)


def test_generate_weight_info_has_real_data(new_roots):
    html_str = r.generate_weight_info(SAMPLE_DATE_STR, SAMPLE_PLACE_ID, SAMPLE_RACE_ID)
    assert 'id="weightInfo"' in html_str
    assert "コース別平均馬体重情報" in html_str
    assert "kg" in html_str


def test_generate_pops_info_has_real_data(new_roots):
    html_str = r.generate_pops_info(SAMPLE_DATE_STR, SAMPLE_PLACE_ID, SAMPLE_RACE_ID)
    assert 'id="popsInfo"' in html_str
    assert "コース別平均人気情報" in html_str
    assert "番人気" in html_str


def test_generate_frame_horse_info_has_real_data(new_roots):
    html_str = r.generate_frame_horse_info(SAMPLE_DATE_STR, SAMPLE_PLACE_ID, SAMPLE_RACE_ID)
    assert 'id="frameHorseInfo"' in html_str
    assert "枠番・馬番 平均情報" in html_str


def test_generate_peds_result_html_has_real_data(new_roots):
    html_str = r.generate_peds_result_html(SAMPLE_DATE_STR, SAMPLE_PLACE_ID, SAMPLE_RACE_ID)
    assert "peds-result-block" in html_str
    assert "血統別成績" in html_str


def test_race_card_breadcrumb_items_includes_same_day_venues_and_races(new_roots):
    """出馬表ページの右側タブ用階層: 同日の他開催場（同じレース番号へリンク）＋
    選択中の開催場の他レースが、それぞれ兄弟として並ぶことを確認する
    （data/race_schedule/race_time_id_list/20241020.csvの実データ:
    04_nigata/05_tokyo/08_kyotoが開催、新潟の1R=202404040601）。
    """
    race_name = "2歳未勝利"
    items = r._race_card_breadcrumb_items(SAMPLE_DATE_STR, "2024/10/20", SAMPLE_PLACE_ID, SAMPLE_RACE_ID, race_name)

    assert items[0] == ("レースカレンダー", "races/index.html")
    assert items[1] == ("2024/10/20", "races/20241020/index.html")

    venue_label, venue_path, venue_siblings = items[2]
    assert venue_label == "新潟"
    assert venue_path == "races/20241020/04_nigataR1.html"
    # 同日に開催している他場（東京・京都）も、同じレース番号(1R)のページへリンクする
    assert ("新潟", "races/20241020/04_nigataR1.html") in venue_siblings
    assert ("東京", "races/20241020/05_tokyoR1.html") in venue_siblings
    assert ("京都", "races/20241020/08_kyotoR1.html") in venue_siblings

    race_label, race_path, race_siblings = items[3]
    assert race_label == f"1R {race_name}"
    assert race_path is None
    # 同じ開催場（新潟）の他レースが兄弟として並ぶ。現在のレースはリンクなし（現在地）
    assert (f"1R {race_name}", None) in race_siblings
    assert ("2R 2歳未勝利", "races/20241020/04_nigataR2.html") in race_siblings
    assert ("12R 3歳以上1勝クラス", "races/20241020/04_nigataR12.html") in race_siblings


def test_race_card_breadcrumb_items_falls_back_when_no_schedule_data(new_roots, monkeypatch):
    """開催スケジュールデータが無い場合でも、自分自身だけの階層を返す（例外にしない）"""
    monkeypatch.setattr(
        r.race_card_dataset_manager, "get_race_time_id_list_df",
        lambda race_day: pd.DataFrame(),
    )

    items = r._race_card_breadcrumb_items(SAMPLE_DATE_STR, "2024/10/20", SAMPLE_PLACE_ID, SAMPLE_RACE_ID, "テストレース")

    assert items[2] == ("新潟", "races/20241020/04_nigataR1.html", [("新潟", "races/20241020/04_nigataR1.html")])
    assert items[3] == ("1R テストレース", None, [("1R テストレース", None)])


def test_make_race_card_html_generates_full_page(new_roots):
    r.make_race_card_html(SAMPLE_DATE_STR, SAMPLE_PLACE_ID, SAMPLE_RACE_ID)

    out_file = new_roots / "public_html" / "races" / SAMPLE_DATE_STR / "04_nigataR1.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_race_card_html({SAMPLE_DATE_STR}, place_id={SAMPLE_PLACE_ID}, race_id={SAMPLE_RACE_ID}) ---")
    print(f"  出力先: {out_file}")
    print(f"  HTML文字数: {len(html_content)}")

    # --- サイト共通ヘッダー・フッター・右側タブ（他ページと統一） ---
    assert '<link rel="stylesheet" href="../../assets/css/styles.css">' in html_content
    assert "pagead2.googlesyndication.com" in html_content
    assert 'rel="icon"' in html_content
    assert "googletagmanager.com/gtag/js?id=G-DNC949064T" in html_content
    assert '<nav class="site-nav">' in html_content
    assert '<aside class="page-calendar-tab">' in html_content
    assert "<footer>" in html_content
    assert "MAR(まーる) 競馬AIデータサイト" in html_content
    assert '<span class="site-brand-name">MAR</span>' in html_content
    # ブレッドクラム（レースカレンダー→その日→このレース）
    assert '<a href="../../races/index.html">レースカレンダー</a>' in html_content
    assert '<a href="../../races/20241020/index.html">2024/10/20</a>' in html_content

    # 旧来の「この日の一覧に戻る」＋前後レースの簡易ナビは、右側タブの階層
    # （同日の開催場＋同開催場の他レース）に統合したため廃止した
    assert '<div class="nav">' not in html_content
    assert "この日の一覧に戻る" not in html_content
    # 右側タブに、同日の他開催場（東京・京都、同じレース番号へリンク）と
    # 選択中の開催場（新潟）の他レースが兄弟として並ぶ
    assert '<a href="../../races/20241020/05_tokyoR1.html">東京</a>' in html_content
    assert '<a href="../../races/20241020/08_kyotoR1.html">京都</a>' in html_content
    assert '<a href="../../races/20241020/04_nigataR2.html">2R 2歳未勝利</a>' in html_content
    assert '<span class="page-calendar-tab-current">1R 2歳未勝利</span>' in html_content
    # ページの先頭へ戻るリンク（ページが長くなったため末尾に追加）
    assert '<a id="pageTop"></a>' in html_content
    assert '<p class="back-to-top"><a href="#pageTop">&uarr; ページの先頭へ戻る</a></p>' in html_content

    # --- 見出し ---
    assert "<h2>2024/10/20 </h2>" in html_content
    assert f"<h2>{NAME_LIST[SAMPLE_PLACE_ID - 1]}競馬場 第1R </h2>" in html_content
    assert "芝2000m 天候:晴 馬場:稍重 クラス:未勝利" in html_content

    # --- 出馬表テーブル（全頭を縦スクロールなしで見られるようtable-wrap--fullを付与） ---
    assert '<div class="table-wrap table-wrap--full">\n  <table id="raceTable">' in html_content

    # --- レース結果・配当 ---
    assert "<h2>レース結果</h2>" in html_content
    assert '<div class="table-wrap table-wrap--full">\n    <table id="resultTable">' in html_content
    assert '<div class="table-wrap table-wrap--full">\n    <table id="payoutTable">' in html_content

    # --- コース別データセクション ---
    assert "コース別平均タイム情報" in html_content
    assert "コース別平均馬体重情報" in html_content
    assert "血統別成績" in html_content
    assert "コース別平均人気情報" in html_content
    assert "枠番・馬番 平均情報" in html_content

    # --- 出走馬詳細レポート ---
    assert '<h2 id="horseReportsSection">出走馬 詳細レポート</h2>' in html_content
    assert '<div id="horseReportsContainer">' in html_content


def test_make_daily_race_card_html_generates_only_available_races(new_roots):
    r.make_daily_race_card_html(SAMPLE_RACE_DAY)

    out_dir = new_roots / "public_html" / "races" / SAMPLE_DATE_STR
    generated = sorted(p.name for p in out_dir.iterdir())

    # race_card CSVを用意したレース（04_nigataR1）のみ生成される
    assert generated == ["04_nigataR1.html"]


def test_make_daily_race_card_html_links_resolve_in_a_single_pass(new_roots, monkeypatch):
    """右側タブの階層は開催スケジュール（race_time_id_list）から組み立てるため、
    生成順に関係なく1回の生成だけで他レースへのリンクが解決されることを確認する
    （旧build_nav_htmlはrace_page_existsで判定していたため2パス生成が必要だった）。
    """
    second_race_id = "202404040602"
    race_card_dir = new_roots / "race_card" / SAMPLE_DATE_STR
    shutil.copy(
        f"data/race_card/{SAMPLE_DATE_STR}/{second_race_id}.csv",
        race_card_dir / f"{second_race_id}.csv",
    )
    race_card_dataset_manager.save_race_info_df(
        pd.DataFrame({"race_type": ["芝"], "course_len": ["2000"], "weather": ["晴"],
                      "ground_state": ["稍重"], "class": ["未勝利"]}),
        SAMPLE_RACE_DAY, second_race_id,
    )

    monkeypatch.setattr(
        r.race_schedule_dataset_manager, "get_daily_id",
        lambda place_id, race_day: [SAMPLE_RACE_ID, second_race_id] if place_id == SAMPLE_PLACE_ID else [],
    )

    r.make_daily_race_card_html(SAMPLE_RACE_DAY)

    out_dir = new_roots / "public_html" / "races" / SAMPLE_DATE_STR
    race1_html = (out_dir / "04_nigataR1.html").read_text(encoding="utf-8")
    race2_html = (out_dir / "04_nigataR2.html").read_text(encoding="utf-8")

    print(f"\n--- make_daily_race_card_html({SAMPLE_RACE_DAY}) 2レースの相互リンク ---")
    print(f"  04_nigataR1.html → 2R: {'04_nigataR2.html' in race1_html}")
    print(f"  04_nigataR2.html → 1R: {'04_nigataR1.html' in race2_html}")

    # 1R→2R（次のレース、1回の生成だけで解決される）
    assert '<a href="../../races/20241020/04_nigataR2.html">' in race1_html
    # 2R→1R（前のレース）
    assert '<a href="../../races/20241020/04_nigataR1.html">' in race2_html


# --- build_table_race_cards（枠順・AI予想Rankの色付け） -------------------------------


def test_build_table_race_cards_colors_waku_and_top_rank():
    df = pd.DataFrame({
        "枠": [1, 3],
        "馬番": [1, 5],
        "馬名": ["サンプルホースA", "サンプルホースB"],
        "性齢": ["牡3", "牝4"],
        "斤量": [56, 54],
        "騎手": ["騎手A", "騎手B"],
        "馬体重(増減)": ["480(+2)", "440(-4)"],
        "score": [0.123, -0.5],
        "rank": [1, 2],
    })

    rows = r.build_table_race_cards(df)

    print("\n--- build_table_race_cards (枠・Rank色付け) ---")
    print(rows)

    # 枠1=白、枠3=赤（WAKU_COLORS）。枠・馬番の両セルに同じ背景色を適用する
    assert rows.count('style="background-color:white; color:#000;"') == 2
    assert rows.count('style="background-color:red; color:#fff;"') == 2
    # AI予想Rank 1位=金色、2位=水色（RANK_COLORS）
    assert '<td style="background-color:#FFD700;">1</td>' in rows
    assert '<td style="background-color:#B0E0E6;">2</td>' in rows


def test_build_table_race_cards_shows_popularity_when_present():
    df = pd.DataFrame({
        "枠": [1, 3],
        "馬番": [1, 5],
        "馬名": ["サンプルホースA", "サンプルホースB"],
        "性齢": ["牡3", "牝4"],
        "斤量": [56, 54],
        "騎手": ["騎手A", "騎手B"],
        "馬体重(増減)": ["480(+2)", "440(-4)"],
        "score": [0.123, -0.5],
        "rank": [1, 2],
        "人気": [1, "**"],
    })

    rows = r.build_table_race_cards(df)

    print("\n--- build_table_race_cards (人気表示) ---")
    print(rows)

    # オッズ確定済みは数値表示、未確定（**等）は「-」表示にする
    assert "<td>1</td>" in rows
    assert "<td>-</td>" in rows


def test_build_table_race_cards_blank_when_waku_undecided():
    df = pd.DataFrame({
        "枠": [None, None],
        "馬番": [None, None],
        "馬名": ["サンプルホースA", "サンプルホースB"],
        "性齢": ["牡3", "牝4"],
        "斤量": [56, 54],
        "騎手": ["騎手A", "騎手B"],
        "馬体重(増減)": ["", ""],
        "score": [pd.NA, pd.NA],
        "rank": [pd.NA, pd.NA],
    })

    rows = r.build_table_race_cards(df)

    # 枠順未確定でも色付け処理自体は落ちず、枠が無いデフォルト色（白）になる
    assert rows.count('style="background-color:#ffffff; color:#000;"') == 4
    assert "<td>サンプルホースA</td>" not in rows  # 馬名はリンク化されるため素のtdにはならない
