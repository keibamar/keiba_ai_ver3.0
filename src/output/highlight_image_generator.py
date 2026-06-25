"""高配当的中ハイライト画像（X投稿用）の生成

的中バッジ・配当額を大きく見せるPNG画像をPillowで組み立てる。Webサイトの
ブランドカラー（src/logic/html_generator/site_nav_html.py周辺で使っているもの
と同じボルドー系）に合わせ、サイトの「顔」として一貫した見た目にする。
画像内には日本語を含むため、Windows標準のメイリオフォントを使う（このプロジェクトは
Windows PC上での運用を前提としているため、フォントファイルをリポジトリに
含めず、システムフォントのパスを直接参照する）。
"""

import os

from PIL import Image, ImageDraw, ImageFont

from src.config import paths

WIDTH, HEIGHT = 1200, 675

COLOR_BG = (122, 36, 56)  # --mar-primary（ボルドー）
COLOR_BG_DARK = (92, 27, 42)  # --mar-primary-dark
COLOR_GOLD = (255, 215, 0)  # 的中・配当の強調色
COLOR_WHITE = (255, 255, 255)
COLOR_LIGHT = (230, 210, 215)

_FONT_BOLD_PATH = r"C:\Windows\Fonts\meiryob.ttc"
_FONT_REGULAR_PATH = r"C:\Windows\Fonts\meiryo.ttc"


def _font(path, size):
    """指定フォント・サイズのImageFontを返す（読み込めない場合はPillow既定フォント）

    既定フォントは日本語が表示できないが、Windows以外の環境で完全に処理が
    落ちるよりは、文字化けしてでも画像自体は生成できるようにするための保険。
    """
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _draw_centered_text(draw, y, text, font, fill, width=WIDTH):
    """指定y座標に、横方向中央揃えでテキストを描画する"""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) / 2, y), text, font=font, fill=fill)


def make_hit_highlight_image(
    bet_type_label, payout, place_name, race_num, race_name, pick_name, out_path,
):
    """高配当的中ハイライト画像を生成し、out_pathに保存する

    Args:
        bet_type_label (str): "単勝"・"複勝"等の式別表示名。
        payout (float): 100円購入あたりの配当額。
        place_name (str): 競馬場名（例: "東京"）。
        race_num (int): レース番号。
        race_name (str): レース名。
        pick_name (str): AI本命馬名。
        out_path (str): 保存先パス（.png）。
    Returns:
        str: 保存したファイルパス（out_pathそのまま）。
    """
    image = Image.new("RGB", (WIDTH, HEIGHT), COLOR_BG)
    draw = ImageDraw.Draw(image)

    # 上下に少し濃い帯を入れて単調なベタ塗りにならないようにする
    draw.rectangle([0, 0, WIDTH, 90], fill=COLOR_BG_DARK)
    draw.rectangle([0, HEIGHT - 70, WIDTH, HEIGHT], fill=COLOR_BG_DARK)

    site_brand_font = _font(_FONT_BOLD_PATH, 36)
    draw.text((40, 22), "MAR(まーる)", font=site_brand_font, fill=COLOR_WHITE)

    hit_font = _font(_FONT_BOLD_PATH, 100)
    _draw_centered_text(draw, 140, "的中！", hit_font, COLOR_GOLD)

    payout_font = _font(_FONT_BOLD_PATH, 80)
    _draw_centered_text(draw, 270, f"{bet_type_label} {payout:.0f}円", payout_font, COLOR_WHITE)

    race_font = _font(_FONT_REGULAR_PATH, 42)
    _draw_centered_text(draw, 400, f"{place_name}{race_num}R {race_name}", race_font, COLOR_LIGHT)

    pick_font = _font(_FONT_BOLD_PATH, 56)
    _draw_centered_text(draw, 465, f"AI本命: {pick_name}", pick_font, COLOR_WHITE)

    footer_font = _font(_FONT_REGULAR_PATH, 28)
    _draw_centered_text(draw, HEIGHT - 56, "競馬AIデータサイト mar-keiba.com", footer_font, COLOR_LIGHT)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path, "PNG")
    return out_path


def highlight_image_path(race_day, race_id, bet_type):
    """指定レース・式別のハイライト画像の保存先パスを返す"""
    folder = os.path.join(paths.HIGHLIGHT_IMAGE_PATH, race_day.strftime("%Y%m%d"))
    return os.path.join(folder, f"{race_id}_{bet_type}.png")
