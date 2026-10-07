"""Claude API を使って記事テキストを生成するクライアント

ANTHROPIC_API_KEY 環境変数（または .env ファイル）から API キーを取得する。
"""

import os

import anthropic

_MODEL = "claude-opus-5"
_MAX_TOKENS = 4096

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY が設定されていません。"
                ".env ファイルか環境変数に設定してください。"
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def generate_article_text(system_prompt: str, user_prompt: str) -> str:
    """Claude API を呼び出して記事テキストを生成する

    Args:
        system_prompt: システムプロンプト（文体・役割指示）
        user_prompt: ユーザープロンプト（記事データと指示）

    Returns:
        生成されたテキスト（Markdown形式）
    """
    client = _get_client()
    message = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
    )
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""
