"""プライバシーポリシー・利用規約ページ（Forge: HTMLFactory）のHTML生成

広告掲載（Google AdSense等）の審査・公開には、個人情報やCookie・広告配信に関する
取り扱いを明記したプライバシーポリシーが必須となる。また、本サイトはnetkeiba等の
第三者サイトから取得したデータを加工・表示しているため、データの出典・利用方針を
利用規約として明文化しておく（第三者サイトの利用規約との関係は法的な確認が必要な
領域のため、公開前に内容を必ず確認・調整すること。ここでは一般的なひな形を提供する）。

他ページと同じsite_nav_html/site_footer_htmlを使い、ヘッダー・フッター・右側タブの
統一を保つ。
"""

from src.logic.html_generator.site_nav_html import (
    SITE_NAME,
    SITE_URL,
    adsense_script_html,
    breadcrumb_html,
    ga4_script_html,
    meta_tags_html,
    site_footer_html,
    site_nav_html,
)
from src.managers import html_manager


def privacy_policy_template():
    """プライバシーポリシーページ（public_html/privacy.html）のHTMLを返す"""
    breadcrumb_items = [("プライバシーポリシー", None)]
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {meta_tags_html(
      f"プライバシーポリシー｜{SITE_NAME}",
      f"{SITE_NAME}のプライバシーポリシー。個人情報・アクセス解析・広告配信における取り扱いについて。",
      f"{SITE_URL}/privacy.html",
  )}
  {adsense_script_html()}
  {ga4_script_html()}
  <title>プライバシーポリシー｜{SITE_NAME}</title>
  <link rel="stylesheet" href="assets/css/styles.css">
  <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
</head>
<body>
  {site_nav_html(base_path="", current_path="privacy.html")}
  {breadcrumb_html(breadcrumb_items, base_path="")}
  <h1>プライバシーポリシー</h1>

  <h2>個人情報の取り扱いについて</h2>
  <p>{SITE_NAME}（以下「当サイト」）では、お問い合わせフォーム等を通じて個人情報をご提供いただく場合がありますが、
  これらは当サイトからの返信や必要な対応以外の目的では利用いたしません。</p>

  <h2>アクセス解析ツールについて</h2>
  <p>当サイトでは、Googleアナリティクス等のアクセス解析ツールを利用しています。このツールはトラフィックデータの収集のためにCookieを使用しています。
  トラフィックデータは匿名で収集されており、個人を特定するものではありません。この機能はCookieを無効にすることで収集を拒否することが可能ですので、
  お使いのブラウザの設定をご確認ください。</p>

  <h2>広告の配信について</h2>
  <p>当サイトは、第三者配信の広告サービス（Google AdSense等）を利用する場合があります。このような広告配信事業者は、
  ユーザーの興味に応じた広告を表示するためCookieを使用することがあります。Cookieを無効にする設定や、
  広告配信事業者によるCookieの使用に関する詳細は、各広告配信事業者のポリシーページをご確認ください。</p>

  <h2>免責事項</h2>
  <p>当サイトに掲載する情報については、できる限り正確な情報を提供するように努めておりますが、正確性や安全性を保証するものではありません。
  情報が古くなっていたり、誤りを含んでいる可能性もございます。当サイトに掲載された内容によって生じた損害等の責任は負いかねますので、
  ご了承ください。</p>

  <h2>プライバシーポリシーの変更について</h2>
  <p>当サイトは、個人情報に関して適用される日本の法令を遵守するとともに、本ポリシーの内容を適宜見直し、その改善に努めます。
  修正された最新のプライバシーポリシーは常に本ページにて開示されます。</p>

  <p><a href="terms.html">利用規約はこちら</a></p>
  <p><a href="./">&larr; HOMEへ戻る</a></p>
  {site_footer_html()}
</body>
</html>
"""


def terms_template():
    """利用規約ページ（public_html/terms.html）のHTMLを返す"""
    breadcrumb_items = [("利用規約", None)]
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {meta_tags_html(
      f"利用規約｜{SITE_NAME}",
      f"{SITE_NAME}の利用規約。掲載データの出典・利用方針について。",
      f"{SITE_URL}/terms.html",
  )}
  {adsense_script_html()}
  {ga4_script_html()}
  <title>利用規約｜{SITE_NAME}</title>
  <link rel="stylesheet" href="assets/css/styles.css">
  <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
</head>
<body>
  {site_nav_html(base_path="", current_path="terms.html")}
  {breadcrumb_html(breadcrumb_items, base_path="")}
  <h1>利用規約</h1>

  <h2>第1条（適用）</h2>
  <p>本規約は、{SITE_NAME}（以下「当サイト」）の利用に関する条件を定めるものです。当サイトをご利用いただく際は、
  本規約に同意したものとみなします。</p>

  <h2>第2条（提供する情報の性質）</h2>
  <p>当サイトが提供する競馬の予想・データ・分析結果等は、すべて参考情報であり、的中や回収を保証するものではありません。
  実際の投票・購入の判断は、利用者ご自身の責任において行ってください。</p>

  <h2>第3条（データの出典について）</h2>
  <p>当サイトに掲載しているレース結果・出馬表・配当等のデータは、JRA（日本中央競馬会）が主催する競馬に関する公開情報を基に、
  第三者の競馬情報サイトを含む各種公開情報源から収集・加工して掲載しています。データの正確性については万全を期しておりますが、
  最新の正確な情報は必ず公式発表（JRA公式サイト等）でご確認ください。</p>
  <p>当サイトに掲載するデータの収集方法・出典の取り扱いについては、関連する第三者サイトの利用規約等を踏まえて
  随時見直しを行います。</p>

  <h2>第4条（禁止事項）</h2>
  <p>利用者は、当サイトの利用にあたり、以下の行為をしてはなりません。</p>
  <ul>
    <li>法令または公序良俗に違反する行為</li>
    <li>当サイトの運営を妨害する行為</li>
    <li>当サイトに掲載された情報を無断で転載・再配布する行為</li>
  </ul>

  <h2>第5条（免責事項）</h2>
  <p>当サイトの情報を利用したことにより生じたいかなる損害についても、当サイトは一切の責任を負いません。</p>

  <h2>第6条（本規約の変更）</h2>
  <p>当サイトは、必要と判断した場合には、利用者に通知することなく本規約を変更することができるものとします。</p>

  <p><a href="privacy.html">プライバシーポリシーはこちら</a></p>
  <p><a href="./">&larr; HOMEへ戻る</a></p>
  {site_footer_html()}
</body>
</html>
"""


def make_privacy_policy_page():
    """public_html/privacy.html を生成する"""
    html_manager.save_privacy_policy_html(privacy_policy_template())


def make_terms_page():
    """public_html/terms.html を生成する"""
    html_manager.save_terms_html(terms_template())
