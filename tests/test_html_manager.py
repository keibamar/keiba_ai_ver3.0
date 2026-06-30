"""src/managers/html_manager.py の add_race_day のテスト

旧 web/src/generators/date_index.py の add_race_day を移植したもので、
public_html/assets/js/raceDays.js（window.racedays）に日付を追加する。
新規作成・追記・重複時のno-op・不正フォーマット時のエラーの各ケースを
新実装単体で検証する（オフライン）。
"""

import json
from datetime import date

import pytest

from src.config import paths
from src.managers import html_manager as new_html_manager

SAMPLE_RACE_DAY = date(2026, 6, 14)

EXISTING_CONTENT = (
    'window.racedays = [\n'
    '  "20241020",\n'
    '  "20241026",\n'
    '  "20260606",\n'
    '  "20260607"\n'
    '];\n'
)


@pytest.fixture
def new_js_path(tmp_path, monkeypatch):
    js_path = tmp_path / "public_html" / "assets" / "js" / "raceDays.js"
    monkeypatch.setattr(new_html_manager, "RACE_DAYS_JS_PATH", str(js_path))
    return js_path


def test_add_race_day_creates_new_file(new_js_path):
    new_js_path.parent.mkdir(parents=True)

    new_html_manager.add_race_day(SAMPLE_RACE_DAY)

    assert new_js_path.read_text(encoding="utf-8") == 'window.racedays = [\n  "20260614"\n];\n'


def test_add_race_day_appends_to_existing_file(new_js_path):
    new_js_path.parent.mkdir(parents=True)
    new_js_path.write_text(EXISTING_CONTENT, encoding="utf-8")

    new_html_manager.add_race_day(SAMPLE_RACE_DAY)

    new_content = new_js_path.read_text(encoding="utf-8")

    print(f"\n--- add_race_day({SAMPLE_RACE_DAY}) ---")
    print(f"  追記前:\n{EXISTING_CONTENT}")
    print(f"  追記後:\n{new_content}")

    assert '"20260614"' in new_content
    assert new_content.endswith('];\n')


def test_add_race_day_noop_if_already_present(new_js_path):
    new_js_path.parent.mkdir(parents=True)
    new_js_path.write_text(EXISTING_CONTENT, encoding="utf-8")

    new_html_manager.add_race_day(date(2026, 6, 6))  # "20260606" は既に登録済み

    assert new_js_path.read_text(encoding="utf-8") == EXISTING_CONTENT


def test_add_race_day_raises_if_array_missing(new_js_path):
    new_js_path.parent.mkdir(parents=True)
    new_js_path.write_text("// no racedays array here\n", encoding="utf-8")

    with pytest.raises(ValueError):
        new_html_manager.add_race_day(SAMPLE_RACE_DAY)


@pytest.fixture
def new_races_root(tmp_path, monkeypatch, new_js_path):
    races_root = tmp_path / "public_html" / "races"
    monkeypatch.setattr(paths, "PUBLIC_HTML_RACES_PATH", str(races_root))
    return races_root


def test_regenerate_race_days_js_only_includes_existing_dirs_from_min_year(new_races_root, new_js_path):
    # 実体のあるディレクトリ（去年・今年・実体の無い日付混在）を用意する
    for day_str in ["20241020", "20260104", "20260620"]:
        (new_races_root / day_str).mkdir(parents=True)
    # raceDays.jsには実体の無い日付（20260103）と去年の日付が残っている状態を再現する
    new_js_path.parent.mkdir(parents=True, exist_ok=True)
    new_js_path.write_text(
        'window.racedays = [\n  "20241020",\n  "20260103",\n  "20260104",\n  "20260620"\n];\n',
        encoding="utf-8",
    )

    new_html_manager.regenerate_race_days_js(min_year=2026)

    content = new_js_path.read_text(encoding="utf-8")
    print(f"\n--- regenerate_race_days_js(min_year=2026) ---\n{content}")

    # 今年（2026年）かつ実体のあるディレクトリのみ残る
    assert '"20260104"' in content
    assert '"20260620"' in content
    # 去年の日付・実体の無い日付（20260103）は含まれない
    assert "20241020" not in content
    assert "20260103" not in content


def test_regenerate_race_days_js_defaults_to_current_year(new_races_root, new_js_path, monkeypatch):
    (new_races_root / "20260104").mkdir(parents=True)
    (new_races_root / "20250101").mkdir(parents=True)

    class _FixedDate(date):
        @classmethod
        def today(cls):
            return date(2026, 6, 23)

    monkeypatch.setattr(new_html_manager, "date", _FixedDate)

    new_html_manager.regenerate_race_days_js()

    content = new_js_path.read_text(encoding="utf-8")
    assert '"20260104"' in content
    assert "20250101" not in content


@pytest.fixture
def new_meetings_js_path(tmp_path, monkeypatch):
    js_path = tmp_path / "public_html" / "assets" / "js" / "raceMeetings.js"
    monkeypatch.setattr(new_html_manager, "RACE_MEETINGS_JS_PATH", str(js_path))
    return js_path


def test_regenerate_race_meetings_js_builds_meetings_from_race_ids(
    new_races_root, new_meetings_js_path, monkeypatch,
):
    import pandas as pd

    day_dir = new_races_root / "20260628"
    day_dir.mkdir(parents=True)
    # 小倉は出馬表ページが生成済み、函館は未生成という状態を再現する
    (day_dir / "10_kokuraR11.html").write_text("dummy", encoding="utf-8")

    def fake_time_id_list_df(race_day):
        assert race_day == date(2026, 6, 28)
        return pd.DataFrame(
            {
                "race_id": ["202602010611", "202610010211"],
                "race_time": ["1540", "1530"],
                "race_name": ["函館記念", "紫川S"],
            }
        )

    monkeypatch.setattr(
        "src.managers.race_card_dataset_manager.get_race_time_id_list_df", fake_time_id_list_df,
    )

    new_html_manager.regenerate_race_meetings_js(min_year=2026)

    content = new_meetings_js_path.read_text(encoding="utf-8")
    print(f"\n--- regenerate_race_meetings_js ---\n{content}")

    assert "window.raceMeetings = " in content
    data = json.loads(content.removeprefix("window.raceMeetings = ").removesuffix(";\n"))
    # race_time昇順（小倉15:30 → 函館15:40）。小倉は出馬表ページが生成済みのため
    # race_card_urlが入り、函館は未生成のためnullになる
    assert data == {
        "20260628": [
            {
                "place_name": "小倉", "race_name": "紫川S", "times": 1, "day_number": 2,
                "race_card_url": "races/20260628/10_kokuraR11.html", "grade": None,
            },
            {
                "place_name": "函館", "race_name": "函館記念", "times": 1, "day_number": 6,
                "race_card_url": None, "grade": None,
            },
        ]
    }


def test_regenerate_race_meetings_js_includes_grade_when_present(
    new_races_root, new_meetings_js_path, monkeypatch,
):
    import pandas as pd

    (new_races_root / "20260628").mkdir(parents=True)

    def fake_time_id_list_df(race_day):
        return pd.DataFrame(
            {
                "race_id": ["202602010611", "202610010211"],
                "race_time": ["1540", "1530"],
                "race_name": ["函館記念", "紫川S"],
                "grade": ["G3", None],
            }
        )

    monkeypatch.setattr(
        "src.managers.race_card_dataset_manager.get_race_time_id_list_df", fake_time_id_list_df,
    )

    new_html_manager.regenerate_race_meetings_js(min_year=2026)

    content = new_meetings_js_path.read_text(encoding="utf-8")
    data = json.loads(content.removeprefix("window.raceMeetings = ").removesuffix(";\n"))

    graded = {m["race_name"]: m["grade"] for m in data["20260628"]}
    assert graded == {"紫川S": None, "函館記念": "G3"}


def test_regenerate_race_meetings_js_skips_days_without_time_id_list(
    new_races_root, new_meetings_js_path, monkeypatch,
):
    import pandas as pd

    (new_races_root / "20260629").mkdir(parents=True)
    monkeypatch.setattr(
        "src.managers.race_card_dataset_manager.get_race_time_id_list_df", lambda race_day: pd.DataFrame(),
    )

    new_html_manager.regenerate_race_meetings_js(min_year=2026)

    content = new_meetings_js_path.read_text(encoding="utf-8")
    assert json.loads(content.removeprefix("window.raceMeetings = ").removesuffix(";\n")) == {}
