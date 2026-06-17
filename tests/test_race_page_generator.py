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
        f"data/RaceCards/{SAMPLE_DATE_STR}/{SAMPLE_RACE_ID}.csv",
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


def test_build_nav_html_structure(new_roots):
    nav_html = r.build_nav_html(SAMPLE_DATE_STR, SAMPLE_PLACE_ID, SAMPLE_RACE_ID)
    assert '<div class="nav">' in nav_html
    assert "この日の一覧に戻る" in nav_html


def test_make_race_card_html_generates_full_page(new_roots):
    r.make_race_card_html(SAMPLE_DATE_STR, SAMPLE_PLACE_ID, SAMPLE_RACE_ID)

    out_file = new_roots / "public_html" / "races" / SAMPLE_DATE_STR / "04_nigataR1.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_race_card_html({SAMPLE_DATE_STR}, place_id={SAMPLE_PLACE_ID}, race_id={SAMPLE_RACE_ID}) ---")
    print(f"  出力先: {out_file}")
    print(f"  HTML文字数: {len(html_content)}")

    # --- ナビゲーション・見出し ---
    assert '<div class="nav">' in html_content
    assert "<h2>2024/10/20 </h2>" in html_content
    assert f"<h2>{NAME_LIST[SAMPLE_PLACE_ID - 1]}競馬場 第1R </h2>" in html_content
    assert "芝2000m 天候:晴 馬場:稍重 クラス:未勝利" in html_content

    # --- 出馬表テーブル ---
    assert '<table id="raceTable">' in html_content

    # --- レース結果・配当 ---
    assert "<h2>レース結果</h2>" in html_content

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
