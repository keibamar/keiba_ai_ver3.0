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

    race_time_id_list（data/race_schedule/race_time_id_list/20241020.csv）と
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

    print(f"\n--- make_daily_index_page({SAMPLE_RACE_DAY}) ---")
    print(f"  出力先: {out_file}")
    print(f"  HTML文字数: {len(html_content)}")
    print(f"  先頭500文字:\n{html_content[:500]}")

    assert "<h1>2024/10/20 レース一覧</h1>" in html_content
    # public_html/races/{date}/index.html から public_html/assets/css/styles.css への正しい相対パス
    assert '<link rel="stylesheet" href="../../assets/css/styles.css">' in html_content
    # サイト共通ヘッダー・フッター・右側タブ（他ページと統一）
    assert '<nav class="site-nav">' in html_content
    assert "pagead2.googlesyndication.com" in html_content
    assert '<aside class="page-calendar-tab">' in html_content
    assert "<footer>" in html_content
    assert '<a href="../../races/index.html">レースカレンダー</a>' in html_content
    assert "<th>新潟競馬場</th>" in html_content
    assert "<th>東京競馬場</th>" in html_content
    assert "<th>京都競馬場</th>" in html_content
    assert "<th>1R</th>" in html_content

    # 前後日のレースページディレクトリが存在しないため、両方disabled
    assert '<span class="disabled">← 前の日</span>' in html_content
    assert '<span class="disabled">→ 次の日</span>' in html_content


def test_make_races_calendar_page_generates_index_html(new_roots, monkeypatch):
    # get_today_main_races_with_courseは出走馬一覧ページのスクレイピングを伴うため、
    # オフラインテストでは固定値に差し替える
    monkeypatch.setattr(
        d.ai_performance_calculator,
        "get_today_main_races_with_course",
        lambda today: [
            {
                "race_id": "202605030611",
                "place_id": 5,
                "race_name": "府中牝馬S",
                "race_time": "1545",
                "race_type": "芝",
                "course_len": 1800,
                "race_day": today,
            }
        ],
    )

    d.make_races_calendar_page()

    out_file = new_roots / "public_html" / "races" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_races_calendar_page() ---")
    print(f"  出力先: {out_file}")
    print(f"  HTML文字数: {len(html_content)}")

    # レースカレンダーのページ全体に専用のサブカラー（section-calendar）を適用する
    assert '<body class="section-calendar">' in html_content
    # サイト共通ヘッダー・右側タブ（他ページと統一）。このページ自身が大きな
    # 月表示カレンダーを持つため、右側タブの小カレンダーは二重表示しない
    assert '<nav class="site-nav">' in html_content
    assert '<aside class="page-calendar-tab">' in html_content
    assert "page-calendar-tab-calendar" not in html_content
    assert "<footer>" in html_content
    # courses/index.html・performance/index.htmlと同様、トップレベルのページでも
    # 横並びのパンくず（HOME > レースカレンダー）を表示する
    assert '<p class="breadcrumb"><a href="../index.html">HOME</a> &rsaquo; <span class="breadcrumb-current">レースカレンダー</span></p>' in html_content
    assert "<h1>開催日カレンダー</h1>" in html_content
    assert '<table id="calendar"></table>' in html_content
    assert 'window.CALENDAR_BASE_PATH = "../";' in html_content
    assert '<script src="../assets/js/raceDays.js"></script>' in html_content
    assert '<script src="../assets/js/calendar.js"></script>' in html_content
    assert '<a href="../index.html">&larr; HOMEへ戻る</a>' in html_content

    # 「本日の開催」に、コース詳細データへのリンクが含まれる
    assert "<h2>本日の開催</h2>" in html_content
    assert '<a href="../courses/05_tokyo/芝-1800.html">東京 <span class="race-type-turf">芝1800m</span></a>' in html_content


def test_calendar_widget_html_uses_given_base_path():
    widget = d.calendar_widget_html(base_path="")

    assert 'window.CALENDAR_BASE_PATH = "";' in widget
    assert '<script src="assets/js/raceDays.js"></script>' in widget
    assert '<script src="assets/js/calendar.js"></script>' in widget
    assert '<table id="calendar"></table>' in widget



def test_today_meetings_html_links_to_race_card_and_course_data(monkeypatch):
    races = [
        {
            "race_id": "202605030611",
            "place_id": 5,
            "race_name": "府中牝馬S",
            "race_time": "1545",
            "race_type": "芝",
            "course_len": 1800,
            "race_day": SAMPLE_RACE_DAY,
        }
    ]
    monkeypatch.setattr(d.html_manager, "race_page_exists", lambda day_str, filename: True)

    html_content = d._today_meetings_html(races, base_path="../")

    print(f"\n--- _today_meetings_html ---\n{html_content}")

    assert '<a href="../races/20241020/05_tokyoR11.html">府中牝馬S</a>' in html_content
    assert '<a href="../courses/05_tokyo/芝-1800.html">東京 <span class="race-type-turf">芝1800m</span></a>' in html_content


def test_today_meetings_html_falls_back_when_race_card_missing(monkeypatch):
    races = [
        {
            "race_id": "202605030611",
            "place_id": 5,
            "race_name": "府中牝馬S",
            "race_time": "1545",
            "race_type": None,
            "course_len": None,
            "race_day": SAMPLE_RACE_DAY,
        }
    ]
    monkeypatch.setattr(d.html_manager, "race_page_exists", lambda day_str, filename: False)

    html_content = d._today_meetings_html(races, base_path="../")

    # 出馬表が無い場合はリンクなしのレース名のみ、コース詳細データは競馬場一覧へリンクする
    assert "<span class=\"main\">府中牝馬S</span>" in html_content
    assert '<a href="../courses/05_tokyo/index.html">東京</a>' in html_content


def test_today_meetings_html_handles_no_races():
    assert "本日開催のレースはありません。" in d._today_meetings_html([])
