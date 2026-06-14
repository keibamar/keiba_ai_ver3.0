"""src/managers/html_manager.py の add_race_day のテスト

旧 web/src/generators/date_index.py の add_race_day を移植したもので、
public_html/assets/js/raceDays.js（window.racedays）に日付を追加する。
新規作成・追記・重複時のno-op・不正フォーマット時のエラーの各ケースを
旧実装と比較しながら検証する（オフライン）。
"""

import os
import sys
from datetime import date

import pytest

from src.managers import html_manager as new_html_manager

# 旧 date_index.add_race_day (web/src/generators/date_index.py) は
# web/src を起点に config/utils をimportするため、web/src をsys.pathに追加する
WEB_SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "src")
if WEB_SRC_PATH not in sys.path:
    sys.path.insert(0, WEB_SRC_PATH)

from generators import date_index as old_date_index  # noqa: E402

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


@pytest.fixture
def old_js_path(tmp_path, monkeypatch):
    js_dir = tmp_path / "old_assets_js"
    monkeypatch.setattr(old_date_index, "JS_FOLDER_PATH", str(js_dir) + os.sep)
    return js_dir / "racedays.js"


def test_add_race_day_creates_new_file_matches_old(new_js_path, old_js_path):
    old_js_path.parent.mkdir(parents=True)

    new_html_manager.add_race_day(SAMPLE_RACE_DAY)
    old_date_index.add_race_day(SAMPLE_RACE_DAY)

    assert new_js_path.read_text(encoding="utf-8") == old_js_path.read_text(encoding="utf-8")
    assert new_js_path.read_text(encoding="utf-8") == 'window.racedays = [\n  "20260614"\n];\n'


def test_add_race_day_appends_to_existing_file_matches_old(new_js_path, old_js_path):
    new_js_path.parent.mkdir(parents=True)
    new_js_path.write_text(EXISTING_CONTENT, encoding="utf-8")

    old_js_path.parent.mkdir(parents=True)
    old_js_path.write_text(EXISTING_CONTENT, encoding="utf-8")

    new_html_manager.add_race_day(SAMPLE_RACE_DAY)
    old_date_index.add_race_day(SAMPLE_RACE_DAY)

    new_content = new_js_path.read_text(encoding="utf-8")
    old_content = old_js_path.read_text(encoding="utf-8")

    assert new_content == old_content
    assert '"20260614"' in new_content
    assert new_content.endswith('];\n')


def test_add_race_day_noop_if_already_present(new_js_path):
    new_js_path.parent.mkdir(parents=True)
    new_js_path.write_text(EXISTING_CONTENT, encoding="utf-8")

    new_html_manager.add_race_day(date(2026, 6, 6))  # "20260606" は既に登録済み

    assert new_js_path.read_text(encoding="utf-8") == EXISTING_CONTENT


def test_add_race_day_raises_if_array_missing(new_js_path, old_js_path):
    new_js_path.parent.mkdir(parents=True)
    new_js_path.write_text("// no racedays array here\n", encoding="utf-8")

    old_js_path.parent.mkdir(parents=True)
    old_js_path.write_text("// no racedays array here\n", encoding="utf-8")

    with pytest.raises(ValueError):
        new_html_manager.add_race_day(SAMPLE_RACE_DAY)

    with pytest.raises(ValueError):
        old_date_index.add_race_day(SAMPLE_RACE_DAY)
