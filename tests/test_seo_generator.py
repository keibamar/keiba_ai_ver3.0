"""src/logic/html_generator/seo_generator.py のテスト（オフライン）。"""

import pytest

from src.config import paths
from src.logic.html_generator import seo_generator as s


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PUBLIC_HTML_PATH", str(tmp_path / "public_html"))
    return tmp_path


def test_robots_txt_content_allows_all_and_links_sitemap():
    content = s.robots_txt_content()

    print(f"\n--- robots_txt_content() ---\n{content}")

    assert "User-agent: *" in content
    assert "Allow: /" in content
    assert "Sitemap: https://mar-keiba.com/sitemap.xml" in content


def test_make_robots_txt_writes_file(new_roots):
    s.make_robots_txt()

    out_file = new_roots / "public_html" / "robots.txt"
    assert out_file.exists()
    assert "Sitemap: https://mar-keiba.com/sitemap.xml" in out_file.read_text(encoding="utf-8")


def test_sitemap_xml_lists_html_pages_with_index_as_directory(new_roots):
    public_html = new_roots / "public_html"
    (public_html / "courses" / "05_tokyo").mkdir(parents=True)
    (public_html / "index.html").write_text("<html></html>", encoding="utf-8")
    (public_html / "courses" / "index.html").write_text("<html></html>", encoding="utf-8")
    (public_html / "courses" / "05_tokyo" / "index.html").write_text("<html></html>", encoding="utf-8")
    (public_html / "courses" / "05_tokyo" / "芝-1400.html").write_text("<html></html>", encoding="utf-8")
    (public_html / "privacy.html").write_text("<html></html>", encoding="utf-8")

    content = s.sitemap_xml_content()

    print(f"\n--- sitemap_xml_content() ---\n{content}")

    assert content.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' in content
    # index.htmlはディレクトリ自体（末尾"/"）のURLにする（/index.htmlという形は出さない）
    assert "<loc>https://mar-keiba.com/</loc>" in content
    assert "<loc>https://mar-keiba.com/courses/</loc>" in content
    assert "<loc>https://mar-keiba.com/courses/05_tokyo/</loc>" in content
    assert "index.html</loc>" not in content
    # index.html以外のページは、そのままファイル名付きのURLになる
    assert "<loc>https://mar-keiba.com/courses/05_tokyo/芝-1400.html</loc>" in content
    assert "<loc>https://mar-keiba.com/privacy.html</loc>" in content


def test_make_sitemap_xml_writes_file(new_roots):
    public_html = new_roots / "public_html"
    public_html.mkdir(parents=True)
    (public_html / "index.html").write_text("<html></html>", encoding="utf-8")

    s.make_sitemap_xml()

    out_file = public_html / "sitemap.xml"
    assert out_file.exists()
    assert "<loc>https://mar-keiba.com/</loc>" in out_file.read_text(encoding="utf-8")
