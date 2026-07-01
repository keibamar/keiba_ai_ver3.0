"""src/logic/html_generator/affiliate_html.py のテスト（オフライン）。"""

import datetime

from src.logic.html_generator import affiliate_html as a


def test_amazon_book_link_html_empty_when_not_configured(monkeypatch):
    monkeypatch.setattr(a, "AMAZON_ASSOCIATE_TAG", "PLACEHOLDER-22")

    assert a.amazon_book_link_html("B000000000", "サンプル書籍") == ""


def test_amazon_book_link_html_builds_link_when_configured(monkeypatch):
    monkeypatch.setattr(a, "AMAZON_ASSOCIATE_TAG", "mar-keiba-22")

    html_out = a.amazon_book_link_html("B000000000", "サンプル書籍")

    assert 'href="https://www.amazon.co.jp/dp/B000000000?tag=mar-keiba-22"' in html_out
    assert 'rel="nofollow noopener sponsored"' in html_out
    assert "サンプル書籍（Amazon）" in html_out


def test_rakuten_book_link_html_empty_when_url_not_given():
    assert a.rakuten_book_link_html(None, "サンプル書籍") == ""
    assert a.rakuten_book_link_html("", "サンプル書籍") == ""


def test_rakuten_book_link_html_builds_link_from_given_url():
    rakuten_url = (
        "https://hb.afl.rakuten.co.jp/ichiba/34d56027.4cd7e36e.34d56028.7817ac3c/"
        "?pc=https%3A%2F%2Fitem.rakuten.co.jp%2Fbook%2F18602254%2F"
    )

    html_out = a.rakuten_book_link_html(rakuten_url, "サンプル書籍")

    assert f'href="{rakuten_url}"' in html_out
    assert "サンプル書籍（楽天市場）" in html_out


def test_book_recommendation_html_empty_when_neither_given(monkeypatch):
    monkeypatch.setattr(a, "AMAZON_ASSOCIATE_TAG", "PLACEHOLDER-22")

    result = a.book_recommendation_html("サンプル書籍", amazon_asin="B000000000", rakuten_url=None)

    assert result == ""


def test_book_recommendation_html_shows_only_configured_links(monkeypatch):
    monkeypatch.setattr(a, "AMAZON_ASSOCIATE_TAG", "mar-keiba-22")

    result = a.book_recommendation_html(
        "サンプル書籍", amazon_asin="B000000000", rakuten_url=None, note="参考図書です",
    )

    assert "サンプル書籍（Amazon）" in result
    assert "楽天市場" not in result
    assert "参考図書です" in result


def test_book_recommendation_html_shows_both_links_when_both_given(monkeypatch):
    monkeypatch.setattr(a, "AMAZON_ASSOCIATE_TAG", "mar-keiba-22")

    result = a.book_recommendation_html(
        "サンプル書籍", amazon_asin="B000000000", rakuten_url="https://hb.afl.rakuten.co.jp/ichiba/x/?pc=y",
    )

    assert "サンプル書籍（Amazon）" in result
    assert "サンプル書籍（楽天市場）" in result


def test_pick_daily_book_is_deterministic_for_same_date():
    target_date = datetime.date(2026, 6, 29)

    first = a.pick_daily_book(target_date)
    second = a.pick_daily_book(target_date)

    assert first == second
    assert first in a.BOOK_CANDIDATES


def test_pick_daily_book_changes_across_many_dates():
    picks = {
        a.pick_daily_book(datetime.date(2026, 1, 1) + datetime.timedelta(days=i))["title"]
        for i in range(30)
    }

    # 30日分見れば候補(5冊)のうち複数が選ばれているはず（全部同じ書籍に固定されない）
    assert len(picks) > 1


def test_daily_book_recommendation_html_uses_picked_book_when_configured(monkeypatch):
    monkeypatch.setattr(a, "AMAZON_ASSOCIATE_TAG", "mar-keiba-22")
    target_date = datetime.date(2026, 6, 29)
    expected_title = a.pick_daily_book(target_date)["title"]

    result = a.daily_book_recommendation_html(target_date)

    assert expected_title in result


def test_a8_service_link_html_empty_when_url_not_given():
    assert a.a8_service_link_html(None, "お名前.com") == ""
    assert a.a8_service_link_html("", "お名前.com") == ""


def test_a8_service_link_html_builds_link_from_given_url():
    url = "https://px.a8.net/svt/ejp?a8mat=XXXXXX+YYYYYY+ZZZZ+1234A"

    html_out = a.a8_service_link_html(url, "お名前.com")

    assert f'href="{url}"' in html_out
    assert 'rel="nofollow noopener sponsored"' in html_out
    assert "お名前.com" in html_out


def test_a8_program_recommendation_html_empty_when_no_programs_configured(monkeypatch):
    monkeypatch.setattr(
        a, "A8_PROGRAM_CANDIDATES",
        [{"name": "お名前.com", "url": None}, {"name": "他社サービス", "url": None}],
    )

    assert a.a8_program_recommendation_html() == ""


def test_a8_program_recommendation_html_lists_only_configured_programs(monkeypatch):
    monkeypatch.setattr(
        a, "A8_PROGRAM_CANDIDATES",
        [
            {"name": "お名前.com", "url": "https://px.a8.net/svt/ejp?a8mat=ONAMAE", "note": "国内シェアNo.1の独自ドメイン取得サービス。"},
            {"name": "他社サービス", "url": None, "note": "未提携のサービス。"},
        ],
    )

    result = a.a8_program_recommendation_html()

    assert "お名前.com" in result
    assert "国内シェアNo.1の独自ドメイン取得サービス。" in result
    assert "他社サービス" not in result
