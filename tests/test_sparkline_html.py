"""src/logic/html_generator/sparkline_html.py のテスト（オフライン）。"""

from src.logic.html_generator import sparkline_html as s


def test_sparkline_svg_renders_polyline_with_points_for_each_value():
    html = s.sparkline_svg([50.0, 60.0, 40.0, 80.0], labels=[2022, 2023, 2024, 2025])

    print(f"\n--- sparkline_svg ---\n{html}")

    assert '<span class="sparkline-wrap">' in html
    assert "<svg" in html
    assert html.count("<circle") == 4
    assert "<polyline points=" in html
    # 最初/最後の値とラベルが表示される
    assert "2022: 50.0" in html
    assert "2025: 80.0" in html


def test_sparkline_svg_returns_empty_string_for_fewer_than_two_points():
    assert s.sparkline_svg([50.0], labels=[2025]) == ""
    assert s.sparkline_svg([], labels=[]) == ""


def test_sparkline_svg_handles_constant_values_without_division_by_zero():
    html = s.sparkline_svg([50.0, 50.0, 50.0], labels=[2023, 2024, 2025])

    assert html.count("<circle") == 3
    assert "2023: 50.0" in html
    assert "2025: 50.0" in html
