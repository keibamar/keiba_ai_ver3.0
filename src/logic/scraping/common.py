"""スクレイピング共通ヘルパー（旧 libs/scraping.py の汎用部分を移植）

各ドメインのスクレイパー（netkeiba_scraper等）から共通利用するHTTP/HTMLチェック関数。
"""

import requests
from bs4 import BeautifulSoup

scraping_header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.3"
}


def url_exists(url: str) -> bool:
    """URLが存在するかどうかを簡易チェックする。

    HEAD を試み、ダメなら GET にフォールバックしてステータスコードを確認する。
    タイムアウトや例外が発生した場合は False を返す。
    """
    try:
        resp = requests.head(url, headers=scraping_header, allow_redirects=True, timeout=5)
        if resp.status_code == 200:
            return True
        resp = requests.get(url, headers=scraping_header, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def fetch_soup(url: str) -> BeautifulSoup:
    """urlを取得し、EUC-JPとしてデコードしたBeautifulSoupを返す"""
    html = requests.get(url, headers=scraping_header)
    html.encoding = "EUC-JP"
    return BeautifulSoup(html.content.decode("euc-jp", "ignore"), "html.parser")


def validate_soup(soup, url: str, func_name: str, require_table: bool = False, selectors: list | None = None) -> bool:
    """soupの中身が期待通りかをチェックする。問題があればログ出力してFalseを返す。

    Args:
        soup: BeautifulSoupオブジェクト
        url (str): チェック対象のURL（ログ用）
        func_name (str): 呼び出し関数名（ログ用）
        require_table (bool): テーブルが必須か
        selectors (list[str]|None): 期待するCSSセレクタのリスト

    Returns:
        bool: 有効ならTrue、無ければFalse
    """
    try:
        if not soup or not getattr(soup, "text", "").strip():
            print(f"{func_name}: empty or no HTML content, skip {url}")
            return False
        if require_table and not soup.find("table"):
            print(f"{func_name}: no <table> found, skip {url}")
            return False
        if selectors:
            for sel in selectors:
                if not soup.select_one(sel):
                    print(f"{func_name}: expected selector '{sel}' not found, skip {url}")
                    return False
        return True
    except Exception as e:
        print(f"{func_name}: validate_soup error {e.__class__.__name__}: {e} for {url}")
        return False


def scraping_error(e):
    """エラー時動作を記載する

    Args:
        e (Exception): エラー内容
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")
