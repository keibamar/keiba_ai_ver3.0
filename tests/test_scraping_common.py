"""src/logic/scraping/common.py のテスト（オフライン）。

fetch_soupがrequests.getに必ずtimeoutを渡すことを検証する。timeoutが無いと、
応答の無いサーバーに対してリクエストが無期限にブロックし続け、多数のレースを
一括処理するバッチで1件がハングするだけで全体が止まってしまうバグがあったため
（実際にこの不具合で長時間ハングする事象が発生した）。
"""

from unittest.mock import MagicMock, patch

from src.logic.scraping import common


def test_fetch_soup_passes_timeout_to_requests_get():
    fake_response = MagicMock()
    fake_response.apparent_encoding = "utf-8"
    fake_response.content = "<html></html>".encode("utf-8")

    with patch("src.logic.scraping.common.requests.get", return_value=fake_response) as mock_get:
        common.fetch_soup("https://example.com")

    assert mock_get.call_args.kwargs["timeout"] == 15


def test_fetch_soup_respects_custom_timeout():
    fake_response = MagicMock()
    fake_response.apparent_encoding = "utf-8"
    fake_response.content = "<html></html>".encode("utf-8")

    with patch("src.logic.scraping.common.requests.get", return_value=fake_response) as mock_get:
        common.fetch_soup("https://example.com", timeout=5)

    assert mock_get.call_args.kwargs["timeout"] == 5
