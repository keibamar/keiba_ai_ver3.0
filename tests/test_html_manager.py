"""src/managers/html_manager.py の add_race_day のテスト

旧 web/src/generators/date_index.py の add_race_day を移植したもので、
public_html/assets/js/raceDays.js（window.racedays）に日付を追加する。
新規作成・追記・重複時のno-op・不正フォーマット時のエラーの各ケースを
新実装単体で検証する（オフライン）。
"""

from datetime import date

import pytest

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
