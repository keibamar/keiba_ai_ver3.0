"""
サンプル: 箇条書き評価 → Claude APIで100〜200文字の短評に変換

実行例:
  python scripts/sample_ai_comment.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.simplefilter("ignore")

import anthropic
from datetime import date
from src.managers import race_card_dataset_manager
from src.logic.html_generator import horse_report_generator
from src.logic.html_generator.horse_report_generator import _horse_comment_html


def extract_sentences_from_html(html: str) -> list[str]:
    """<ul><li>...</li></ul> から箇条書きテキストを取り出す"""
    import re
    return re.findall(r"<li>(.*?)</li>", html, re.DOTALL)


def generate_short_comment(
    horse_name: str,
    sentences: list[str],
    race_info: str = "",
    style: str = "個人ブログ",
) -> str:
    """
    箇条書き評価ポイントをもとに短評テキストを生成する。

    Args:
        horse_name: 馬名
        sentences: 評価ポイントのリスト
        race_info: レース概要（任意）
        style: 文体の指示（"個人ブログ" など）

    Returns:
        100〜200文字の短評テキスト
    """
    if not sentences:
        return ""

    bullet_text = "\n".join(f"- {s}" for s in sentences)

    system_prompt = (
        "あなたは競馬歴20年以上の個人ブロガーです。"
        "データと独自の視点を組み合わせた率直な短評を書くスタイルで知られています。"
        "文体の特徴："
        "・「〜だろう」「〜とみる」「〜が焦点」など断定的・主観的な語尾を使う"
        "・箇条書きや体言止めは使わず、続き物の文章で書く"
        "・難しい競馬用語は使ってもよいが、読みやすく簡潔に"
        "・ポジティブ/ネガティブ両方を1〜2文にまとめ、最後に自分の結論を1文入れる"
    )

    user_prompt = (
        f"以下は「{horse_name}」の評価ポイントです。{race_info}\n\n"
        f"{bullet_text}\n\n"
        "これらを踏まえて、100〜200文字の短評を書いてください。"
        "箇条書きの内容をすべて盛り込む必要はなく、最も重要なポイントを自然な文章でまとめてください。"
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY が設定されていません。\n"
            ".env に ANTHROPIC_API_KEY=sk-ant-... を追加してください。\n"
            "取得先: https://console.anthropic.com/"
        )
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
    )
    return response.content[0].text.strip()


def main():
    # ── サンプルデータ取得 ───────────────────────────────────────────────
    race_day = date(2026, 7, 19)
    race_id = "202602011201"  # 函館R1

    card = race_card_dataset_manager.get_race_cards(race_day, race_id)
    if card.empty:
        print("race_card が見つかりませんでした")
        return

    horse_name_col = next((c for c in card.columns if "名" in c), None)
    if horse_name_col is None:
        print("馬名列が見つかりませんでした")
        return

    # 最初の3頭でデモ
    for i, (_, row) in enumerate(card.iterrows()):
        if i >= 3:
            break

        horse_name = str(row.get(horse_name_col, "")).strip()
        if not horse_name:
            continue

        print(f"\n{'='*60}")
        print(f"■ {horse_name}（馬番{row.get('馬番', '?')}）")

        # horse_report 取得
        report = horse_report_generator.build_horse_report(horse_name, 2, race_id, "20260719")
        if not report:
            print("  → レポートなし")
            continue

        # 指数取得
        try:
            score = float(row.get("score", 50))
            rank_val = int(row.get("rank", 5))
            popularity = int(row.get("人気", row.get("pop", 5)))
            idx_mar = float(row.get("idx_mar")) if row.get("idx_mar") not in (None, "") else None
            rank_mar = int(float(row.get("rank_mar"))) if row.get("rank_mar") not in (None, "") else None
        except Exception:
            score, rank_val, popularity = 50, 5, 5
            idx_mar, rank_mar = None, None

        # 箇条書き生成
        comment_html = _horse_comment_html(
            report, score, rank_val, popularity,
            idx_mar=idx_mar, rank_mar=rank_mar,
        )
        sentences = extract_sentences_from_html(comment_html)

        if not sentences:
            print("  → 評価ポイントなし（中程度の評価で言及なし）")
            continue

        print("\n【箇条書き（現行）】")
        for s in sentences:
            print(f"  ・{s}")

        print("\n【Claude生成短評】")
        race_info = "函館芝1200m 新馬戦"
        short_comment = generate_short_comment(horse_name, sentences, race_info=f"（{race_info}）")
        print(f"  {short_comment}")
        print(f"  文字数: {len(short_comment)}")


if __name__ == "__main__":
    main()
