"""傾向分析テキスト生成（Claude API使用）

trend_analyzer.get_day_stats / get_week_stats の結果を受け取り、
Claude Haiku を使って自然言語の傾向分析コメントを生成する。

毎回 API を呼ぶと費用がかかるため、生成済みのテキストは
data/trend/ にキャッシュする。
"""

import json
import os
from datetime import date, timedelta

import anthropic

from src.config import paths

# テキストキャッシュディレクトリ
TREND_DATA_PATH = os.path.join(paths.DATA_PATH, "trend")
DAILY_CACHE_DIR = os.path.join(TREND_DATA_PATH, "daily")
WEEKLY_CACHE_DIR = os.path.join(TREND_DATA_PATH, "weekly")

for _d in [TREND_DATA_PATH, DAILY_CACHE_DIR, WEEKLY_CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

# 開催場名マップ（place_id → 日本語名）
PLACE_NAMES = {
    "1": "札幌", "2": "函館", "3": "福島", "4": "新潟",
    "5": "東京", "6": "中山", "7": "中京", "8": "京都",
    "9": "阪神", "10": "小倉",
}

# 馬場状態の遷移テキスト
def _ground_change_label(prev: str, curr: str) -> str:
    rank = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
    if prev not in rank or curr not in rank:
        return ""
    diff = rank[curr] - rank[prev]
    if diff > 0:
        return f"（{prev}→{curr}、悪化）"
    elif diff < 0:
        return f"（{prev}→{curr}、改善）"
    return f"（{prev}で変わらず）"


def _format_stats_for_prompt(stats: dict, prev_day_stats: dict = None, prev_week_stats: dict = None) -> str:
    """統計dictを Claude への入力テキストに変換する"""
    lines = []
    lines.append(f"【日付】{stats['date'][:4]}年{stats['date'][4:6]}月{stats['date'][6:8]}日")

    # 開催場別情報（芝/ダートごとに上り3F・荒れ度を含む）
    lines.append("\n【開催場別情報】")
    for place, info in stats.get("ground_by_place", {}).items():
        ground = info.get("ground_state", "不明")
        weather = info.get("weather", "不明")
        prev_ground = ""
        if prev_week_stats and place in prev_week_stats.get("ground_by_place", {}):
            pg = prev_week_stats["ground_by_place"][place]["ground_state"]
            prev_ground = _ground_change_label(pg, ground)
        elif prev_day_stats and place in prev_day_stats.get("ground_by_place", {}):
            pg = prev_day_stats["ground_by_place"][place]["ground_state"]
            prev_ground = _ground_change_label(pg, ground)
        lines.append(f"\n  ■ {place}  天気={weather} 馬場={ground}{prev_ground}  計{info.get('race_count',0)}R")

        # 種別ごとの詳細
        for rt, cs in info.get("course_stats", {}).items():
            up = f"上り3F平均{cs['up3f']}秒" if cs.get("up3f") else "上り3Fデータなし"
            # 上り速度コメント
            if cs.get("up3f"):
                u = cs["up3f"]
                if rt == "芝":
                    speed = "速め" if u <= 34.5 else ("やや速め" if u <= 35.2 else ("標準" if u <= 35.8 else "遅め"))
                else:
                    speed = "速め" if u <= 38.0 else ("標準" if u <= 39.5 else "遅め")
                up += f"（{speed}）"
            upset_note = f"波乱{cs['upset_count']}件/{cs['race_count']}R" if cs.get("race_count") else ""
            lines.append(f"    {rt}: {up}  {upset_note}")

        # 前週同場・上り3F比較
        prev_ref = prev_week_stats or prev_day_stats
        if prev_ref and place in prev_ref.get("ground_by_place", {}):
            ref_label = "前週比" if prev_week_stats else "前日比"
            prev_cs = prev_ref["ground_by_place"][place].get("course_stats", {})
            for rt, cs in info.get("course_stats", {}).items():
                if rt in prev_cs and cs.get("up3f") and prev_cs[rt].get("up3f"):
                    diff = round(cs["up3f"] - prev_cs[rt]["up3f"], 2)
                    sign = "+" if diff > 0 else ""
                    lines.append(f"      ↑{ref_label}（{rt}上り3F）: {sign}{diff}秒")

    # 走破タイム（勝ち時計）前週比較（同クラス・同馬場のみ）
    win_times = stats.get("win_times", {})
    prev_ref = prev_week_stats or prev_day_stats
    prev_ref_label = "前週" if prev_week_stats else "前日"
    prev_win_times = (prev_ref or {}).get("win_times", {})

    def fmt_sec(s):
        m, sec = divmod(s, 60)
        return f"{int(m)}:{sec:04.1f}" if m else f"{sec:.1f}秒"

    if win_times and prev_win_times:
        lines.append("\n【勝ち時計の傾向（同クラス・同馬場での前週比較）】")
        found_any = False
        for place, rt_dict in win_times.items():
            if place not in prev_win_times:
                continue
            for rt, dist_dict in rt_dict.items():
                if rt not in prev_win_times[place]:
                    continue
                for dist, cg_dict in dist_dict.items():
                    prev_dist = prev_win_times[place].get(rt, {}).get(dist, {})
                    for cg_key, info in cg_dict.items():
                        if cg_key not in prev_dist:
                            continue
                        prev_info = prev_dist[cg_key]
                        diff = round(info["avg_sec"] - prev_info["avg_sec"], 2)
                        sign = "+" if diff > 0 else ""
                        direction = "遅くなっている" if diff > 0.5 else ("速くなっている" if diff < -0.5 else "ほぼ同水準")
                        cls = info["class"]
                        ground = info["ground"]
                        lines.append(
                            f"  {place} {rt}{dist}m [{cls}・馬場:{ground}]: "
                            f"今週{fmt_sec(info['avg_sec'])} / {prev_ref_label}{fmt_sec(prev_info['avg_sec'])} "
                            f"({sign}{diff}秒 → {direction})"
                        )
                        found_any = True
        if not found_any:
            lines.append("  ※ 同クラス・同距離・同馬場の前週比較データなし")

    # 全体荒れ度
    upset = stats.get("upset", {})
    lines.append(f"\n【全体荒れ度】{upset.get('label','不明')}（10倍超勝利: {upset.get('high_odds_count',0)}/{upset.get('race_count',0)}R）")

    # 全体上り3F（参考）
    up3f = stats.get("up3f", {})
    if up3f:
        up_parts = [f"{rt}{t}秒" for rt, t in up3f.items()]
        lines.append(f"【全体上り3F平均】{' / '.join(up_parts)}")

    # 勝ち馬・3着内の傾向
    horse_tend = stats.get("horse_tendencies", {})
    winner_tend = horse_tend.get("winner", {})
    top3_tend = horse_tend.get("top3", {})
    if winner_tend or top3_tend:
        lines.append("\n【勝ち馬・3着内の傾向】")
        if winner_tend.get("front_rate") is not None:
            w_front = winner_tend["front_rate"] * 100
            w_closer = winner_tend["closer_rate"] * 100
            w_pop = winner_tend.get("avg_popularity")
            pop_str = f" / 平均{w_pop:.1f}番人気" if w_pop else ""
            lines.append(f"  勝ち馬（{winner_tend.get('valid_races',0)}R）: 前残り{w_front:.0f}% / 差し追込{w_closer:.0f}%{pop_str}")
        if top3_tend.get("front_rate") is not None:
            t_front = top3_tend["front_rate"] * 100
            t_closer = top3_tend["closer_rate"] * 100
            t_pop = top3_tend.get("avg_popularity")
            pop_str = f" / 平均{t_pop:.1f}番人気" if t_pop else ""
            lines.append(f"  3着内馬（{top3_tend.get('valid_races',0)}頭）: 前残り{t_front:.0f}% / 差し追込{t_closer:.0f}%{pop_str}")

    # AI成績
    ai = stats.get("ai_perf", {})
    if ai:
        lines.append(f"\n【AI予想成績（MAR）】")
        if ai.get("win_hit") is not None:
            lines.append(f"  単勝: 的中率{ai['win_hit']*100:.1f}% / 回収率{ai.get('win_return',0):.1f}%")
        if ai.get("place_hit") is not None:
            lines.append(f"  複勝: 的中率{ai['place_hit']*100:.1f}% / 回収率{ai.get('place_return',0):.1f}%")
        if ai.get("trio_box_hit") is not None:
            lines.append(f"  3連複5頭BOX: 的中率{ai['trio_box_hit']*100:.1f}% / 回収率{ai.get('trio_box_return',0):.1f}%")

    return "\n".join(lines)


def _format_weekly_for_prompt(week_stats: dict, prev_week_stats: dict = None) -> str:
    """週次統計dictを Claude への入力テキストに変換する"""
    lines = []
    sat = week_stats.get("sat", {})
    sun = week_stats.get("sun", {})
    combined = week_stats.get("combined", {})

    lines.append(f"【週次データ】{week_stats['sat_date'][:4]}年{week_stats['sat_date'][4:6]}月{week_stats['sat_date'][6:8]}日（土）〜{week_stats['sun_date'][6:8]}日（日）")
    lines.append("\n--- 土曜 ---")
    lines.append(_format_stats_for_prompt(sat))
    lines.append("\n--- 日曜 ---")
    lines.append(_format_stats_for_prompt(sun, prev_day_stats=sat))

    # 展開傾向
    pace = combined.get("pace_bias", {})
    if pace:
        front_pct = pace.get("front_rate", 0) * 100
        closer_pct = pace.get("closer_rate", 0) * 100
        total = pace.get("valid_races", 0)
        lines.append(f"\n【展開傾向（土日合算 {total}R）】")
        lines.append(f"  前残り（4角3番手以内）: {front_pct:.1f}%")
        lines.append(f"  差し・追込（4角6番手以降）: {closer_pct:.1f}%")

    # 土日合算の勝ち馬・3着内の傾向
    horse_tend = combined.get("horse_tendencies", {})
    winner_tend = horse_tend.get("winner", {})
    top3_tend = horse_tend.get("top3", {})
    if winner_tend or top3_tend:
        lines.append("\n【土日合算：勝ち馬・3着内の傾向】")
        if winner_tend.get("front_rate") is not None:
            w_front = winner_tend["front_rate"] * 100
            w_closer = winner_tend["closer_rate"] * 100
            w_pop = winner_tend.get("avg_popularity")
            pop_str = f" / 平均{w_pop:.1f}番人気" if w_pop else ""
            lines.append(f"  勝ち馬（{winner_tend.get('valid_races',0)}R）: 前残り{w_front:.0f}% / 差し追込{w_closer:.0f}%{pop_str}")
        if top3_tend.get("front_rate") is not None:
            t_front = top3_tend["front_rate"] * 100
            t_closer = top3_tend["closer_rate"] * 100
            t_pop = top3_tend.get("avg_popularity")
            pop_str = f" / 平均{t_pop:.1f}番人気" if t_pop else ""
            lines.append(f"  3着内馬（{top3_tend.get('valid_races',0)}頭）: 前残り{t_front:.0f}% / 差し追込{t_closer:.0f}%{pop_str}")

    # 前週比較（展開・馬傾向・荒れ度）
    if prev_week_stats:
        lines.append("\n--- 前週との比較 ---")
        prev_combined = prev_week_stats.get("combined", {})

        # 荒れ度
        prev_upset = prev_combined.get("upset", {})
        curr_upset = combined.get("upset", {})
        if prev_upset and curr_upset:
            diff = curr_upset.get("upset_ratio", 0) - prev_upset.get("upset_ratio", 0)
            sign = "+" if diff > 0 else ""
            lines.append(f"  荒れ度: {prev_upset.get('label')}→{curr_upset.get('label')} （{sign}{diff*100:.1f}pt変化）")

        # 展開傾向
        prev_pace = prev_combined.get("pace_bias", {})
        curr_pace = combined.get("pace_bias", {})
        if prev_pace and curr_pace:
            prev_front = prev_pace.get("front_rate", 0) * 100
            curr_front = curr_pace.get("front_rate", 0) * 100
            diff_front = curr_front - prev_front
            sign = "+" if diff_front > 0 else ""
            lines.append(f"  前残り率: {prev_front:.1f}%→{curr_front:.1f}% （{sign}{diff_front:.1f}pt変化）")
            prev_closer = prev_pace.get("closer_rate", 0) * 100
            curr_closer = curr_pace.get("closer_rate", 0) * 100
            diff_closer = curr_closer - prev_closer
            sign = "+" if diff_closer > 0 else ""
            lines.append(f"  差し追込率: {prev_closer:.1f}%→{curr_closer:.1f}% （{sign}{diff_closer:.1f}pt変化）")

        # 勝ち馬・3着内傾向
        prev_horse = prev_combined.get("horse_tendencies", {})
        curr_horse = combined.get("horse_tendencies", {})
        prev_winner = prev_horse.get("winner", {})
        curr_winner = curr_horse.get("winner", {})
        if prev_winner and curr_winner:
            prev_pop = prev_winner.get("avg_popularity")
            curr_pop = curr_winner.get("avg_popularity")
            if prev_pop and curr_pop:
                diff_pop = curr_pop - prev_pop
                sign = "+" if diff_pop > 0 else ""
                lines.append(f"  勝ち馬平均人気: {prev_pop:.1f}番→{curr_pop:.1f}番 （{sign}{diff_pop:.1f}）")
        prev_top3 = prev_horse.get("top3", {})
        curr_top3 = curr_horse.get("top3", {})
        if prev_top3 and curr_top3:
            prev_t3pop = prev_top3.get("avg_popularity")
            curr_t3pop = curr_top3.get("avg_popularity")
            if prev_t3pop and curr_t3pop:
                diff = curr_t3pop - prev_t3pop
                sign = "+" if diff > 0 else ""
                lines.append(f"  3着内平均人気: {prev_t3pop:.1f}番→{curr_t3pop:.1f}番 （{sign}{diff:.1f}）")

    return "\n".join(lines)


def _call_claude(prompt: str) -> str:
    """Claude Haiku でテキスト生成（500〜1000字目安）"""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # .env から読む
        env_path = os.path.join(paths.PROJECT_ROOT, ".env")
        if os.path.isfile(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("ANTHROPIC_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY が未設定です。.env に設定してください。")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=6000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def generate_daily_comment(stats: dict, prev_day_stats: dict = None, prev_week_stats: dict = None) -> str:
    """日次傾向分析コメントを生成する（500〜1000字）

    Args:
        stats: get_day_stats() の返り値
        prev_day_stats: 前日の stats（土→日のみ使用）
        prev_week_stats: 前週同曜日の stats

    Returns:
        str: 生成されたコメントテキスト
    """
    date_key = stats.get("date", "")
    cache_path = os.path.join(DAILY_CACHE_DIR, f"{date_key}.json")

    # キャッシュチェック
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        return cached.get("text", "")

    data_text = _format_stats_for_prompt(stats, prev_day_stats, prev_week_stats)

    prompt = f"""あなたはJRA競馬の傾向分析を書くライターです。
以下のデータを元に、その日の競馬傾向の短評を日本語で500〜1000字程度で書いてください。

【文体の指示】
- 競馬ファンが読む日記・コラムのイメージ。砕けすぎず、でも堅苦しくもない自然な文体で
- 「〜でした」「〜でしたね」「〜といったところ」など読みやすい語尾を使う
- 専門用語はそのまま使ってOK（馬場・稍重・上り3Fなど）
- 太字（**）やMarkdownは使わない
- 箇条書きは使わず、読み物として流れる文章に

【競馬の基礎知識（文中で必要に応じて言及してください）】
- 上り3F：最後の600mのタイム。芝では34秒台が速め、35秒台が普通、36秒超が遅め。ダートでは38秒台が速め、39秒台が普通
- 勝ち時計：1着馬の走破タイム（コース全体のタイム）。前週同条件と比べて速い/遅いで馬場傾向を判断できる
- 雨と馬場の関係（重要）：
  ・芝は雨が降ると馬場が緩んで走りにくくなり、タイムが遅くなりやすく、荒れやすくなる傾向がある
  ・ダートは適度な雨で砂が締まり、走りやすくなってタイムが速くなることがある（稍重・重が高速化のサイン）
  ・ただし雨が降りすぎると芝もダートも走りにくくなり、タイムが極端に遅くなることがある
  ・逆に良馬場でも日照りが続いた場合、芝は力がいる馬場になり、ダートはパサパサになってタイムが遅くなることもある
  ・馬場状態（良/稍重/重/不良）と天気を組み合わせて、タイムの速い/遅いの背景を推測してください

【構成の指示】
1. 冒頭で当日の開催全体の天気・馬場をひとことで概観する
2. 開催場ごとにセクションを分け（見出し: 「■ 競馬場名」）、各場で以下を書く：
   - 天気・馬場状態（前日または前週から変化があれば触れる）
   - 芝とダートそれぞれについて：
     ・上り3Fの速さと、その意味（速ければ瞬発力勝負、遅ければスタミナ戦になりやすいなど）
     ・勝ち時計データがある場合は前週比で速くなった/遅くなったかを言及する
     ・天気・馬場状態と組み合わせて、なぜそのタイムになったかを推測する（雨で芝が遅化、雨でダートが速化など）
   - その場での荒れ具合（何Rで波乱があったかを自然な文章で）
3. 最後に全体まとめとして：
   - 全体の荒れ傾向の評価
   - AI予想（MAR）の成績評価（回収率100%以上が黒字ライン）
   - 今週末に向けた一言
文字数は多少多くなっても構いません。丁寧に書いてください。

【データ】
{data_text}
"""

    text = _call_claude(prompt)

    # キャッシュ保存
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"date": date_key, "text": text}, f, ensure_ascii=False, indent=2)

    return text


def generate_weekly_comment(week_stats: dict, prev_week_stats: dict = None,
                             next_weekend_hint: str = "") -> str:
    """週次振り返りコメントを生成する（1500〜3000字）

    Args:
        week_stats: get_week_stats() の返り値
        prev_week_stats: 前週の week_stats
        next_weekend_hint: 来週の天気・コース変更などのヒント（ユーザー提供）

    Returns:
        str: 生成されたコメントテキスト
    """
    date_key = week_stats.get("sat_date", "")
    cache_path = os.path.join(WEEKLY_CACHE_DIR, f"{date_key}.json")

    if os.path.isfile(cache_path) and not next_weekend_hint:
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        return cached.get("text", "")

    data_text = _format_weekly_for_prompt(week_stats, prev_week_stats)

    next_hint_section = ""
    if next_weekend_hint:
        next_hint_section = f"\n\n【来週末情報（ユーザー提供）】\n{next_weekend_hint}"

    prompt = f"""あなたはJRA競馬の傾向分析専門家です。
以下の土日2日間のデータを元に、週次振り返りレポートを日本語で作成してください。

【競馬の基礎知識（参考）】
- 上り3F：最後の600mのタイム。芝では34秒台が速め、35秒台が普通、36秒超が遅め。ダートでは38秒台が速め、39秒台が普通
- 勝ち時計：前週同条件と比べて速い/遅いで馬場傾向を判断できる
- 馬場と天気の関係：芝は雨で緩みやすく、ダートは適度な雨で締まりやすい。ただし例外も多い

【執筆で絶対に守ること（重要）】

❶ 一般論を発見のように書かない
  - 「前有利になりやすい」「雨でダートが速くなる」などは競馬の常識。これを今週の特徴として書いても意味がない
  - 重要なのは「今週は前週や平均と比べてどうだったか」。データが一般論通りなら通り、例外なら例外として評価する

❷ 全開催場の傾向まとめは基本的に書かない
  - 各競馬場は場所・コース・使用条件が異なるため、「全体として前有利だった」などのまとめは情報が薄い
  - 例外は「今週は全3場に共通する特徴があった」と言える場合のみ（例：「全場で荒れ度が前週比で増加」「全場で差し馬の台頭が目立った」など）

❸ 前週比較を分析の軸にする
  - データには前週との比較が含まれる。これを使って「先週と比べて何が変わったか」を競馬場ごとに語ること
  - 前週データがない場合は、各場の土曜→日曜の変化を中心に語る

❹ 勝ち馬・3着内の比較でオリジナリティを出す
  - 勝ち馬の傾向と3着内の傾向の差を語ること
  - 「勝つこと」と「上位入着すること」の違いを浮き彫りにできると深みが出る

【構成（必ずこの順で書くこと）】

### 1. 今週のポイント（冒頭3行以内）
「今週何が特徴的だったか」を一言で。前週と比べて何が変わったか・変わらなかったかを含めてよい

### 2. 開催場ごとの馬場・傾向分析
各開催場について：
- 土曜→日曜の馬場状態の変化、勝ち時計・上り3Fの動き
- 前週と比べてどうだったか（データがあれば必ず言及）
- その場特有の傾向（例：「札幌は前週比で差し馬の比率が増えた」）
- 各場の荒れ度（波乱件数）も触れる

### 3. 勝ち馬・3着内の傾向（最重要）
- 今週の勝ち馬にはどんな特徴があったか（前残り率・平均人気の前週比を使うこと）
- 勝ち馬と3着内馬の傾向の差を語る
  → 「勝ち馬は〇〇傾向だったが、3着内には〇〇な馬も含まれ、差がある/ない」
- 土曜と日曜で傾向が変化していれば言及

### 4. AI予想（MAR）成績
土曜・日曜それぞれの的中率・回収率。良かった条件・悪かった条件があれば具体的に

### 5. 来週末の展望
今週の傾向が続きそうか変化しそうか。天気・コース変更の情報があれば使う

【文体・フォーマット】
- 読みやすい文章形式。各セクションに見出し付き（Markdown可）
- 「〜でした」「〜といったところ」など自然な語尾
- 総文字数1500〜2500字。コンパクトに、でも密度濃く

【データ】
{data_text}{next_hint_section}
"""

    text = _call_claude(prompt)

    if not next_weekend_hint:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"sat_date": date_key, "text": text}, f, ensure_ascii=False, indent=2)

    return text
