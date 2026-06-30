"""アフィリエイトリンク（Amazonアソシエイト・楽天アフィリエイト・A8.net）のHTML生成

AdSense（site_nav_html.AD_SLOT_*）と同じ考え方で、Amazonは登録完了前はプレース
ホルダーのタグのままにしておき、実際のタグに差し替えるまでリンクは表示しない
（プレースホルダーのまま壊れたリンクを出してしまわないようにするため）。

楽天は、Amazonの「タグ1つを商品ASINに付けるだけ」という方式と異なり、
「リンク作成」ツールで発行されるリンクIDが商品が出品されているショップ単位で
変わる（例: 楽天ブックス出品分と中古書店出品分で別のID）ことが実際に確認できた
ため、共通IDから組み立てるのではなく、ツールで生成した完全な紹介URLを
BOOK_CANDIDATESの各書籍にそのまま保持する方式にしている。

書籍の具体的な紹介先（ASIN・楽天の完全な紹介URL）はBOOK_CANDIDATESで管理する。

A8.netは楽天と同様、提携プログラムごとに発行される完全なトラッキングURL
（https://px.a8.net/svt/ejp?a8mat=...）をそのまま保持する方式（ASP共通の
リダイレクト形式で、商品コードだけを差し替えて組み立てられる単純な構造では
ないため）。A8_PROGRAM_CANDIDATESで管理し、未提携（URLがNoneのまま）の
プログラムは自動的に表示から省かれる。競馬関連に限らず、A8.netで承認された
プログラム全般（ドメイン登録等のサイト運営者向けサービスを含む）を対象にする。
"""

import datetime
import html
import random

# Amazonアソシエイトのトラッキング ID。
AMAZON_ASSOCIATE_TAG = "markeiba-22"

# 紹介する書籍の候補（血統・データ分析系）。rakuten_urlは楽天アフィリエイトの
# 「リンク作成」ツールで発行された完全な紹介URL（hb.afl.rakuten.co.jp/...）を
# そのまま入れる。未生成の書籍はrakuten_urlをNoneにしておき、Amazonリンクのみ
# 表示する（book_recommendation_htmlが自動で対応）。
BOOK_CANDIDATES = [
    {
        "title": "亀谷敬正の競馬血統辞典2",
        "amazon_asin": "4801490867",
        "rakuten_url": (
            "https://hb.afl.rakuten.co.jp/ichiba/34d56027.4cd7e36e.34d56028.7817ac3c/"
            "?pc=https%3A%2F%2Fitem.rakuten.co.jp%2Fbook%2F18602254%2F"
        ),
        "note": "血統分析の定番シリーズ最新作。AI予想でも使っている血統データの読み解き方の参考に。",
    },
    {
        "title": "亀谷敬正の競馬血統辞典",
        "amazon_asin": "4801490646",
        "rakuten_url": (
            "https://hb.afl.rakuten.co.jp/ichiba/5556cac6.a0533177.5556cac7.0a45b379/"
            "?pc=https%3A%2F%2Fitem.rakuten.co.jp%2Fvaboo%2Fva0646149480u20%2F"
        ),
        "note": "血統分析の定番シリーズ第1作。",
    },
    {
        "title": "勝ち馬がわかる 血統の教科書2.0",
        "amazon_asin": "4262144739",
        "rakuten_url": (
            "https://hb.afl.rakuten.co.jp/ichiba/34d56027.4cd7e36e.34d56028.7817ac3c/"
            "?pc=https%3A%2F%2Fitem.rakuten.co.jp%2Fbook%2F17411932%2F"
        ),
        "note": "血統の基礎から実践的な予想方法まで解説した入門書。",
    },
    {
        "title": "競馬力を上げる馬券統計学の教科書",
        "amazon_asin": "4865358633",
        "rakuten_url": (
            "https://hb.afl.rakuten.co.jp/ichiba/5556cb6e.528eef31.5556cb6f.fd56d27f/"
            "?pc=https%3A%2F%2Fitem.rakuten.co.jp%2Frakutenkobo-ebooks%2F"
            "e1449116eec638858cceb5ba4bc4231c%2F"
        ),
        "note": "データ・統計の視点から馬券戦略を見直したい人向け。",
    },
    {
        "title": "東大式「確率統計」で競馬に勝つ方法",
        "amazon_asin": "4584104115",
        "rakuten_url": (
            "https://hb.afl.rakuten.co.jp/ichiba/5556cac6.a0533177.5556cac7.0a45b379/"
            "?pc=https%3A%2F%2Fitem.rakuten.co.jp%2Fvaboo%2Fva4115410458u30%2F"
        ),
        "note": "確率統計の基礎からオリジナル指数づくりまでを解説。",
    },
    {
        "title": "競馬の教科書 発想を変えるだけで回収率は上がる",
        "amazon_asin": "4801490719",
        "rakuten_url": (
            "https://hb.afl.rakuten.co.jp/ichiba/5556cb6e.528eef31.5556cb6f.fd56d27f/"
            "?pc=https%3A%2F%2Fitem.rakuten.co.jp%2Frakutenkobo-ebooks%2F"
            "f5665ec99d8d39df8143d22e0ef75be3%2F"
        ),
        "note": "能力比較と馬場読みに絞った発想転換で回収率を上げる考え方。",
    },
    {
        "title": "競馬で勝ち続ける1%の人になる方法",
        "amazon_asin": "4881442457",
        "rakuten_url": (
            "https://hb.afl.rakuten.co.jp/ichiba/5556cb6e.528eef31.5556cb6f.fd56d27f/"
            "?pc=https%3A%2F%2Fitem.rakuten.co.jp%2Frakutenkobo-ebooks%2F"
            "9d7beb8bebeb31ac80b053e6424b5106%2F"
        ),
        "note": "資金管理・期待値の考え方を含めた馬券戦略の入門書。",
    },
]

# A8.net提携プログラムの候補。urlはA8.netの「リンク作成」ツールで発行された
# 完全なトラッキングURL（https://px.a8.net/svt/ejp?a8mat=...）。提携申請・承認前、
# または良い候補が見つかっていない間はurlをNoneにしておき、
# a8_program_recommendation_htmlの表示から自動的に省かれる。
# オッズパーク・楽天競馬はA8.netの提携プログラム一覧で見当たらず（2026/6時点）、
# 「競馬最強の法則WEB」（A8.netで実在を確認）は評判が賛否両論のため見送った。
A8_PROGRAM_CANDIDATES = [
    {
        "name": "お名前.com",
        "url": "https://px.a8.net/svt/ejp?a8mat=4B684D+EPADKI+50+2HGLCX",
        "note": "国内シェアNo.1の独自ドメイン取得サービス。",
    },
]


def _is_amazon_configured():
    """Amazonアソシエイトの登録（タグ差し替え）が完了しているか"""
    return bool(AMAZON_ASSOCIATE_TAG) and AMAZON_ASSOCIATE_TAG != "PLACEHOLDER-22"


def amazon_book_link_html(asin, title):
    """Amazonアソシエイトの書籍リンクHTMLを返す（未登録時は空文字列）

    Args:
        asin (str): AmazonのASIN（商品コード）
        title (str): 表示するリンクテキスト（書籍名）

    Returns:
        str: アンカータグのHTML（未登録時は空文字列）
    """
    if not _is_amazon_configured():
        return ""
    url = f"https://www.amazon.co.jp/dp/{asin}?tag={AMAZON_ASSOCIATE_TAG}"
    title_esc = html.escape(str(title))
    return (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer sponsored" '
        f'class="affiliate-link affiliate-link--amazon">{title_esc}（Amazon）</a>'
    )


def rakuten_book_link_html(rakuten_url, title):
    """楽天アフィリエイトの書籍リンクHTMLを返す（rakuten_url未指定時は空文字列）

    Args:
        rakuten_url (str): 楽天アフィリエイトの「リンク作成」ツールで発行された
            完全な紹介URL（BOOK_CANDIDATESのrakuten_urlを参照）。
        title (str): 表示するリンクテキスト（書籍名）

    Returns:
        str: アンカータグのHTML（rakuten_urlが空の場合は空文字列）
    """
    if not rakuten_url:
        return ""
    url_esc = html.escape(str(rakuten_url))
    title_esc = html.escape(str(title))
    return (
        f'<a href="{url_esc}" target="_blank" rel="noopener noreferrer sponsored" '
        f'class="affiliate-link affiliate-link--rakuten">{title_esc}（楽天市場）</a>'
    )


def book_recommendation_html(title, amazon_asin=None, rakuten_url=None, note=""):
    """書籍紹介ウィジェットのHTMLを返す

    Amazonアソシエイトが未登録、または楽天アフィリエイトのrakuten_urlが
    未指定の間は、それぞれのリンクが空文字列になり自動的に省略される。
    どちらも無い場合は空文字列を返す（リンクの枠だけ表示してしまわないようにするため）。

    Args:
        title (str): 書籍名
        amazon_asin (str | None): AmazonのASIN
        rakuten_url (str | None): 楽天アフィリエイトの完全な紹介URL
        note (str): 紹介文（任意）

    Returns:
        str: 書籍紹介ウィジェットのHTML（リンクが1つも無ければ空文字列）
    """
    links = []
    if amazon_asin:
        link = amazon_book_link_html(amazon_asin, title)
        if link:
            links.append(link)
    if rakuten_url:
        link = rakuten_book_link_html(rakuten_url, title)
        if link:
            links.append(link)
    if not links:
        return ""

    note_html = f'<p class="book-recommendation-note">{html.escape(note)}</p>' if note else ""
    links_html = " / ".join(links)
    return f"""<div class="book-recommendation">
      <p class="book-recommendation-title">📚 {html.escape(str(title))}</p>
      {note_html}
      <p class="book-recommendation-links">{links_html}</p>
    </div>"""


def pick_daily_book(target_date=None):
    """BOOK_CANDIDATESから当日分の書籍を1つ選ぶ（日付をシードに決定的に選択）

    同じ日に何度サイトを再生成しても同じ書籍になり、日付が変われば別の書籍に
    ランダムに切り替わる（クライアント側でのランダム表示ではなく、生成時点で
    1つに確定する静的サイト生成の方針に合わせている）。

    Args:
        target_date (datetime.date | None): 対象日（省略時は本日）

    Returns:
        dict: BOOK_CANDIDATESの要素のうち1つ
    """
    target_date = target_date or datetime.date.today()
    rng = random.Random(target_date.isoformat())
    return rng.choice(BOOK_CANDIDATES)


def daily_book_recommendation_html(target_date=None):
    """当日分の書籍紹介ウィジェットのHTMLを返す（pick_daily_book + book_recommendation_html）

    Args:
        target_date (datetime.date | None): 対象日（省略時は本日）

    Returns:
        str: 書籍紹介ウィジェットのHTML
    """
    book = pick_daily_book(target_date)
    return book_recommendation_html(
        book["title"],
        amazon_asin=book.get("amazon_asin"),
        rakuten_url=book.get("rakuten_url"),
        note=book.get("note", ""),
    )


def a8_service_link_html(url, name):
    """A8.net提携プログラムのリンクHTMLを返す（url未指定時は空文字列）

    Args:
        url (str): A8.netの「リンク作成」ツールで発行された完全なトラッキングURL。
        name (str): 表示するリンクテキスト（サービス名）。

    Returns:
        str: アンカータグのHTML（urlが空の場合は空文字列）
    """
    if not url:
        return ""
    url_esc = html.escape(str(url))
    name_esc = html.escape(str(name))
    return (
        f'<a href="{url_esc}" target="_blank" rel="noopener noreferrer sponsored" '
        f'class="affiliate-link affiliate-link--a8">{name_esc}</a>'
    )


def a8_program_recommendation_html():
    """A8.net提携プログラム紹介ウィジェットのHTMLを返す（提携済みのプログラムのみ表示）

    書籍紹介（daily_book_recommendation_html）と異なり日替わりにはせず、
    A8_PROGRAM_CANDIDATESのうち提携済み（urlが設定済み）のプログラムを
    すべてまとめて表示する（候補数が少なく、1つに絞るより並べて見せる方が
    自然なため）。

    Returns:
        str: A8.net提携プログラム紹介ウィジェットのHTML（提携済みプログラムが無ければ空文字列）
    """
    items = []
    for program in A8_PROGRAM_CANDIDATES:
        link = a8_service_link_html(program.get("url"), program["name"])
        if link:
            items.append((link, program.get("note", "")))
    if not items:
        return ""

    rows = "".join(
        f'<p class="a8-program-item">{link}'
        + (f' <span class="a8-program-note">{html.escape(note)}</span>' if note else "")
        + "</p>"
        for link, note in items
    )
    return f"""<div class="a8-program-recommendation">
      <p class="a8-program-title">🔗 おすすめサービス</p>
      {rows}
    </div>"""
