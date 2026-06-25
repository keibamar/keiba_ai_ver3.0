"""src/logic/html_generator/rate_gauge_html.py のテスト（オフライン）。

的中率ゲージの色グラデーション（低=青→高=赤）と、回収率ゲージの100%基準線・
色分け（100%以下=青、100%超=橙、200%超=赤）を検証する。
"""

from src.logic.html_generator import rate_gauge_html as g


def test_hit_rate_gauge_width_matches_value():
    html = g.hit_rate_gauge_html(13.0)
    assert "width: 13.0%" in html
    assert "13.0%</span>" in html


def test_hit_rate_gauge_color_is_low_at_zero_and_high_at_max():
    low_html = g.hit_rate_gauge_html(0.0)
    high_html = g.hit_rate_gauge_html(100.0)

    print(f"\n--- hit_rate_gauge_html(0.0) ---\n{low_html}")
    print(f"--- hit_rate_gauge_html(100.0) ---\n{high_html}")

    assert "rgb(31,79,214)" in low_html
    # 50%以上はグラデーション上限（赤）でクランプされる
    assert "rgb(214,47,47)" in high_html


def test_return_rate_gauge_clamps_width_but_shows_real_value():
    html = g.return_rate_gauge_html(523.9)

    print(f"\n--- return_rate_gauge_html(523.9) ---\n{html}")

    assert 'width: 100.0%' in html
    assert "523.9%</span>" in html


def test_return_rate_gauge_width_is_half_at_100_percent():
    # ゲージ右端=200%なので、100%（損益分岐点）はゲージ幅の半分の位置になる
    html = g.return_rate_gauge_html(100.0)
    assert "width: 50.0%" in html


def test_return_rate_gauge_color_thresholds():
    blue_html = g.return_rate_gauge_html(99.9)
    orange_html = g.return_rate_gauge_html(150.0)
    red_html = g.return_rate_gauge_html(250.0)

    assert "#1f4fd6" in blue_html
    assert "#e07b00" in orange_html
    assert "#cc2222" in red_html


def test_return_rate_big_html_renders_value_with_same_color_thresholds():
    html = g.return_rate_big_html(150.0)

    print(f"\n--- return_rate_big_html(150.0) ---\n{html}")

    assert html == '<span class="rate-big" style="color: #e07b00;">150.0%</span>'
    assert "#1f4fd6" in g.return_rate_big_html(99.9)
    assert "#cc2222" in g.return_rate_big_html(250.0)


def test_return_rate_color_matches_gauge_thresholds():
    assert g.return_rate_color(99.9) == "#1f4fd6"
    assert g.return_rate_color(150.0) == "#e07b00"
    assert g.return_rate_color(250.0) == "#cc2222"


def test_bet_result_cell_html_shows_payout_instead_of_rate_when_hit():
    html_content = g.bet_result_cell_html(True, 510.0)

    assert '<span class="hit-badge win">的中</span>' in html_content
    assert "510円" in html_content
    # 的中率(%)としては表示しない
    assert "%" not in html_content


def test_bet_result_cell_html_shows_only_miss_badge_when_not_hit():
    html_content = g.bet_result_cell_html(False, None)

    assert html_content == '<span class="hit-badge miss">不的中</span>'


def test_bet_result_cell_html_emphasizes_big_payout():
    html_content = g.bet_result_cell_html(True, 1500.0)

    assert '<span class="hit-badge win hit-badge-big">的中</span>' in html_content
    assert '<span class="hit-payout-big">1500円</span>' in html_content


def test_bet_result_cell_html_does_not_emphasize_below_threshold():
    html_content = g.bet_result_cell_html(True, 999.0)

    assert "hit-badge-big" not in html_content
    assert "hit-payout-big" not in html_content


def test_bet_result_cell_html_shows_dash_when_void():
    assert g.bet_result_cell_html(False, None, void=True) == '<span class="hit-badge void">-</span>'
    # 的中していても、void=True（本命馬除外等）なら的中バッジより優先して「-」を出す
    assert g.bet_result_cell_html(True, 510.0, void=True) == '<span class="hit-badge void">-</span>'
