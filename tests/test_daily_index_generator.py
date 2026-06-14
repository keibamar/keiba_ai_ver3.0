from datetime import date

import pytest

from src.config import paths
from src.logic.html_generator import daily_index_generator as d
from src.managers import race_schedule_dataset_manager

SAMPLE_DATE_STR = "20241020"
SAMPLE_RACE_DAY = date(2024, 10, 20)


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    """public_htmlの出力先をtmp_path配下に切り替える。

    race_time_id_list（texts/race_calendar/race_time_id_list/20241020.csv）と
    race_schedule（data/race_schedule）は実データをそのまま参照する。
    """
    monkeypatch.setattr(paths, "PUBLIC_HTML_RACES_PATH", str(tmp_path / "public_html" / "races"))
    return tmp_path


def test_load_race_info_returns_real_race_names(new_roots):
    info = d.load_race_info(SAMPLE_DATE_STR)
    assert "202404040601" in info
    assert info["202404040601"]["race_time"] == "0950"
    assert info["202404040601"]["race_name"]


def test_group_place_races_and_build_table_rows(new_roots):
    race_info_dict = d.load_race_info(SAMPLE_DATE_STR)
    race_id_list = race_schedule_dataset_manager.get_daily_id(0, SAMPLE_RACE_DAY)
    files_info_list = [
        {"place_id": int(str(rid)[4:6]), "race_num": int(str(rid)[-2:]), "file": rid}
        for rid in race_id_list
    ]

    place_races = d.group_place_races(files_info_list, race_info_dict)
    assert set(place_races.keys()) == {"04_nigata", "05_tokyo", "08_kyoto"}
    assert place_races["04_nigata"]["display"] == "新潟"
    assert len(place_races["04_nigata"]["races"]) == 12

    table_rows, place_keys = d.build_table_rows(place_races, SAMPLE_DATE_STR)
    # PLACE_LIST順（04_nigata, 05_tokyo, 08_kyoto）
    assert place_keys == ["04_nigata", "05_tokyo", "08_kyoto"]
    assert table_rows.count("<tr>") == 12
    assert "<th>1R</th>" in table_rows
    # まだレースページが生成されていないため <a href> は出ない
    assert "<a href=" not in table_rows


def test_make_daily_index_page_generates_index_html(new_roots):
    d.make_daily_index_page(SAMPLE_RACE_DAY)

    out_file = new_roots / "public_html" / "races" / SAMPLE_DATE_STR / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    assert "<h1>2024/10/20 レース一覧</h1>" in html_content
    assert "<th>新潟競馬場</th>" in html_content
    assert "<th>東京競馬場</th>" in html_content
    assert "<th>京都競馬場</th>" in html_content
    assert "<th>1R</th>" in html_content

    # 前後日のレースページディレクトリが存在しないため、両方disabled
    assert '<span class="disabled">← 前の日</span>' in html_content
    assert '<span class="disabled">→ 次の日</span>' in html_content
