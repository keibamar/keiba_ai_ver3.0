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
from src.managers import ai_performance_dataset_manager, html_manager


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


def _about_ai_stats_html():
    """about.html用のAI通算成績サマリーHTMLを返す（動的生成）"""
    try:
        df = ai_performance_dataset_manager.get_ai_performance_dataset()
        if df.empty:
            return ""
        perf = ai_performance_dataset_manager.aggregate(df)
        win = perf.get("win", {})
        place = perf.get("place", {})
        n = win.get("n", 0)
        win_hit = win.get("hit_rate")
        win_roi = win.get("return_rate")
        place_hit = place.get("hit_rate")
        place_roi = place.get("return_rate")
        if n < 10 or win_hit is None:
            return ""
        years_df = df[df["year"].notna()]
        year_min = int(years_df["year"].min()) if not years_df.empty else 2020
        year_max = int(years_df["year"].max()) if not years_df.empty else 2025
        return f"""<div class="about-ai-stats">
  <h3>AIの現在の通算成績（{year_min}〜{year_max}年、{n}R集計）</h3>
  <table>
    <thead><tr><th>券種</th><th>的中率</th><th>回収率</th></tr></thead>
    <tbody>
      <tr><td>単勝</td><td>{win_hit:.1f}%</td><td>{win_roi:.0f}%</td></tr>
      <tr><td>複勝</td><td>{place_hit:.1f}%</td><td>{place_roi:.0f}%</td></tr>
    </tbody>
  </table>
  <p class="about-ai-stats-note">成績は毎週更新しています。詳細は<a href="performance/index.html">AI成績レポート</a>をご覧ください。</p>
</div>"""
    except Exception:
        return ""


def about_template():
    """このサイトについて（about.html）のHTMLを返す"""
    breadcrumb_items = [("このサイトについて", None)]
    ai_stats_html = _about_ai_stats_html()
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {meta_tags_html(
      f"このサイトについて｜{SITE_NAME}",
      f"{SITE_NAME}（まーる）は、機械学習AIを活用した競馬予想・コース別データ分析サイトです。AIの仕組みや運営方針をご紹介します。",
      f"{SITE_URL}/about.html",
  )}
  {adsense_script_html()}
  {ga4_script_html()}
  <title>このサイトについて｜{SITE_NAME}</title>
  <link rel="stylesheet" href="assets/css/styles.css">
  <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
</head>
<body>
  {site_nav_html(base_path="", current_path="about.html")}
  {breadcrumb_html(breadcrumb_items, base_path="")}
  <main>
  <h1>このサイトについて</h1>

  <h2>MARとは</h2>
  <p class="about-mar-concept">「<strong>血統データ</strong>と<strong>走破時計</strong>から勝ち馬を導く」競馬予想AI</p>
  <p>MAR（まーる）は、機械学習AIを活用した競馬予想・データ分析サイトです。
  <strong>血統（父・母父・祖父系）</strong>と<strong>過去の走破時計・コース適性</strong>を核に、
  JRAの過去レースデータで学習したAIが各馬をスコアリングし、予想順位を提供します。
  AIによる予想成績（的中率・回収率）や、競馬場・コース別の傾向データをすべて無料で公開しています。</p>

  <div class="about-content-grid">
    <div class="about-content-card">
      <div class="about-content-icon">📅</div>
      <strong>AI予想出馬表</strong>
      <p>毎週土日のレースにAIスコア・予想順位を掲載。MAR / MAR-hit / MAR-val の3モデル指数を表示。</p>
      <a href="races/">出馬表を見る &rarr;</a>
    </div>
    <div class="about-content-card">
      <div class="about-content-icon">📊</div>
      <strong>AI成績レポート</strong>
      <p>単勝・複勝・三連複の的中率・回収率を週次・競馬場別・コース別に公開。</p>
      <a href="performance/">成績を見る &rarr;</a>
    </div>
    <div class="about-content-card">
      <div class="about-content-icon">🏟️</div>
      <strong>コース別データ</strong>
      <p>各競馬場・距離・芝ダート別の傾向データ（枠番・血統・馬場・通過順など）を掲載。</p>
      <a href="courses/">データを見る &rarr;</a>
    </div>
  </div>

  <h2>運営者について</h2>
  <div class="about-author-box">
    <p>メーカーでソフトウェアエンジニアとして勤務しながら、趣味の競馬をより深く楽しむために
    データ分析・AI開発・サイト運営を行っています。
    AI予想の精度向上に向けて、特徴量の改善やモデルのチューニングを日々続けています。</p>
    <p><strong>キタサンブラック</strong>をきっかけに競馬を始め、
    <strong>アーモンドアイ</strong>の天皇賞を現地で観戦。
    スタンドを揺るがす歓声と、他馬を圧倒するその走りに感動し、競馬にすっかりはまりました。
    データと現地観戦、両方で競馬を楽しんでいます。</p>
  </div>

  <h2>AIの仕組み</h2>
  <p>当サイトのAIは、<strong>LightGBM（勾配ブースティング）</strong>をベースに構築しています。
  2020年〜2025年のJRAレース結果（全10競馬場・全コース）を学習データとして使用し、
  100以上の特徴量から各馬のスコアを算出しています。</p>
  <p>MARのコアとなるのは<strong>血統データ</strong>と<strong>走破時計</strong>の2軸です。
  そこに騎手・枠順・オッズなどの補助データを加えることで、より精度の高い予想を実現しています。</p>

  <div class="about-feature-section">
    <h4 class="about-feature-group-label">コアデータ（メイン）</h4>
    <ul class="about-feature-badges">
      <li><strong>血統情報</strong>：父・母父・父方祖父の3世代</li>
      <li><strong>走破時計・過去成績</strong>：直近5走のタイム・コース適性・着順</li>
    </ul>
    <h4 class="about-feature-group-label">補助データ（サブ）</h4>
    <ul class="about-feature-badges about-feature-badges--sub">
      <li><strong>騎手成績</strong>：当該コースの勝率・複勝率</li>
      <li><strong>枠順・馬番</strong>：コースごとの傾向</li>
      <li><strong>オッズ</strong>：市場情報（回収率最適化に活用）</li>
      <li><strong>ペース適性</strong>：上り3Fタイムのコース平均との差</li>
      <li><strong>馬体重</strong>：前走からの増減・絶対値</li>
    </ul>
  </div>

  <p>スコアが高い馬ほど「このコース・この条件で好走しやすい」と判断された馬です。
  ただし競馬は不確定要素を多く含み、スコアが高くても必ず勝つわけではありません。</p>

  <h3>MARモデルファミリー</h3>
  <p>当サイトでは、学習データ・目的関数の異なる3種類のモデルを <strong>MARファミリー</strong> として運用しています。
  それぞれ「当レースのオッズ・人気情報をどう使うか」が異なり、的中率・回収率のバランスが変わります。</p>

  <div class="about-model-grid">
    <div class="about-model-card about-model-card-main">
      <div class="about-model-card-name">
        <span class="model-tag model-tag-main">MAR</span>
        <strong>メインモデル</strong>
      </div>
      <p>AIスコアと市場オッズを組み合わせたブレンドモデル。
      的中率・回収率のバランスが最も安定しており、当サイトの看板指数です。</p>
      <ul class="about-model-detail">
        <li>2026年実績：単勝29.9% / 回収率167%</li>
        <li>表示：<strong>レース当日</strong>、オッズ確定後に出馬表へ表示</li>
      </ul>
    </div>
    <div class="about-model-card about-model-card-hit">
      <div class="about-model-card-name">
        <span class="model-tag model-tag-hit">MAR-hit</span>
        <strong>的中率重視</strong>
      </div>
      <p>当レースのオッズ・人気を特徴量として<strong>使わない</strong>モデル。
      血統・走破時計・騎手成績など「馬の実力」だけでスコアリングするため、
      市場の評価に左右されない純粋な能力予測になります。
      3モデル中、単勝的中率が最も高くなっています。</p>
      <ul class="about-model-detail">
        <li>2026年実績：単勝32.6% / 回収率128%</li>
        <li>表示：<strong>前日</strong>から出馬表に表示（人気・オッズ不要のため）</li>
      </ul>
    </div>
    <div class="about-model-card about-model-card-val">
      <div class="about-model-card-name">
        <span class="model-tag model-tag-val">MAR-val</span>
        <strong>回収率重視</strong>
      </div>
      <p>当レースの<strong>確定人気を特徴量として使用</strong>し、さらに学習ラベルに
      「高配当での好走ほど高評価」というオッズボーナスを付与して学習したモデル。
      「AIが高評価しているのに市場人気が低い馬＝市場が見落としている穴馬」を
      重点的に本命選択することで、高い回収率を狙います。
      3モデル中、単勝回収率が最も高くなっています。</p>
      <ul class="about-model-detail">
        <li>2026年実績：単勝29.5% / 回収率192%</li>
        <li>表示：<strong>レース当日</strong>、確定人気が出た後に出馬表へ表示</li>
      </ul>
    </div>
  </div>

  {ai_stats_html}

  <h2>モデルの改善について</h2>
  <p>AIモデルは継続的に改善しており、特徴量の追加・目的関数の変更・ハイパーパラメータの最適化を
  定期的に行っています。新旧モデルは同じ直近レースデータで比較評価し、
  成績が改善したと確認できた場合のみ本番環境へ切り替えています。
  現在は <strong>MARモデルファミリー（MAR / MAR-hit / MAR-val）</strong> を本番運用しています。</p>

  <h2>運営方針</h2>
  <div class="about-policy-box">
    <p>当サイトはAI予想の成績データをすべてオープンに公開しています。
    成績がよい期間だけを選んで掲載するようなことはせず、良い時も悪い時も実績のまま掲載します。</p>
    <p>AIの改善・更新を行った際は、新旧モデルの成績を比較した上で採用モデルを決定しています。
    「AIを使えば必ず儲かる」という印象を与えることは意図していません。
    データを根拠に自分自身で考えるための参考情報として、透明性の高い形で公開することが目的です。</p>
  </div>

  <h2>お問い合わせ</h2>
  <div class="about-contact-box">
    <p>ご意見・ご質問は、X（旧Twitter）のDMよりお気軽にどうぞ。</p>
    <a href="https://x.com/keiba_mar" target="_blank" rel="noopener noreferrer">&#120143; X: @keiba_mar</a>
  </div>

  <p style="margin-top:28px;"><a href="privacy.html">プライバシーポリシー</a>　<a href="terms.html">利用規約</a></p>
  <p><a href="./">&larr; HOMEへ戻る</a></p>
  </main>
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


def make_about_page():
    """public_html/about.html を生成する"""
    html_manager.save_about_html(about_template())
