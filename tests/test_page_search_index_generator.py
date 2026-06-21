"""src/logic/html_generator/page_search_index_generator.py のテスト（オフライン）。"""

import json

from src.config import paths
from src.logic.html_generator import page_search_index_generator as g


def test_build_page_search_index_includes_venues_and_courses():
    entries = g.build_page_search_index()

    print(f"\n--- build_page_search_index()（先頭5件） ---\n{entries[:5]}")

    labels = [e["label"] for e in entries]
    paths_ = [e["path"] for e in entries]

    assert "東京 コース詳細データ" in labels
    assert "東京 AI成績" in labels
    assert "courses/05_tokyo/index.html" in paths_
    assert "performance/course/05_tokyo/index.html" in paths_
    # コース別エントリも含まれる
    assert any(label.startswith("東京 芝1400m") for label in labels)
    assert "courses/05_tokyo/芝-1400.html" in paths_
    assert "performance/course/05_tokyo/芝-1400.html" in paths_


def test_make_page_search_index_js_writes_valid_js_data_file(tmp_path, monkeypatch):
    out_path = tmp_path / "assets" / "js" / "page-search-index.js"
    monkeypatch.setattr(g, "PAGE_SEARCH_INDEX_JS_PATH", str(out_path))

    g.make_page_search_index_js()

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")

    print(f"\n--- page-search-index.js（先頭300文字） ---\n{content[:300]}")

    assert content.startswith("const PAGE_SEARCH_INDEX = ")
    json_part = content[len("const PAGE_SEARCH_INDEX = ") : -2]
    parsed = json.loads(json_part)
    assert isinstance(parsed, list)
    assert {"label": "東京 コース詳細データ", "path": "courses/05_tokyo/index.html"} in parsed
