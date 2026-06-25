"""src/output/highlight_image_generator.py のテスト（オフライン）。

画像の見た目（ピクセル単位の検証）はテストできないため、ファイルが正しく
生成されること・パスの組み立てが正しいことのみ検証する。
"""

from datetime import date

from src.config import paths
from src.output import highlight_image_generator as h


def test_make_hit_highlight_image_creates_png_file(tmp_path):
    out_path = str(tmp_path / "20241020" / "202404040609_win.png")

    result = h.make_hit_highlight_image(
        "単勝", 1000.0, "新潟", 9, "十日町特別", "ヴァンヴィーヴ", out_path,
    )

    print(f"\n--- make_hit_highlight_image ---\n{result}")

    assert result == out_path
    assert (tmp_path / "20241020" / "202404040609_win.png").exists()
    # PNGとして正しく読み込めることを確認する（壊れたファイルでないこと）
    from PIL import Image
    with Image.open(out_path) as img:
        assert img.size == (h.WIDTH, h.HEIGHT)


def test_highlight_image_path_builds_expected_path(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "HIGHLIGHT_IMAGE_PATH", str(tmp_path / "highlight_images"))

    result = h.highlight_image_path(date(2024, 10, 20), "202404040609", "win")

    assert result == str(tmp_path / "highlight_images" / "20241020" / "202404040609_win.png")
