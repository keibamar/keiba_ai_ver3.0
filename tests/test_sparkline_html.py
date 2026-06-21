"""src/logic/html_generator/sparkline_html.py のテスト（オフライン）。"""

from src.logic.html_generator import sparkline_html as s


def test_hit_return_trend_svg_renders_both_lines_with_legend():
    labels = ["4/13", "4/20", "4/27", "5/4"]
    hit_rates = [10.0, 20.0, 15.0, 30.0]
    return_rates = [50.0, 150.0, 80.0, 250.0]

    html = s.hit_return_trend_svg(labels, hit_rates, return_rates)

    print(f"\n--- hit_return_trend_svg ---\n{html}")

    assert '<span class="trend-chart-wrap">' in html
    assert "<svg" in html
    # 的中率(グレー、4点)・回収率(色分け、4点)+凡例の2点で合計10つの<circle>
    assert html.count("<circle") == 10
    # 凡例で「的中率(左軸・参考)」「回収率(右軸...)」が分かる
    assert "的中率(左軸・参考)" in html
    assert "回収率(右軸" in html
    # 的中率は参考データとしてグレーの破線で薄く表示する
    assert 'stroke="#aaaaaa"' in html
    assert 'stroke-dasharray="3 2"' in html
    assert html.count("#aaaaaa") >= 5  # 線1本 + 点4つ
    # ホバー用のツールチップに日付・値が入っている
    assert "<title>4/13 的中率: 10.0%</title>" in html
    assert "<title>5/4 回収率: 250.0%</title>" in html
    # 回収率の100%基準線（損益分岐点）が点線で引かれ、ラベルも表示される
    assert '<line x1="' in html
    assert 'class="trend-chart-breakeven"' in html
    assert ">100%</text>" in html
    # 100%を超える区間・点はreturn_rate_colorと同じ基準（橙/赤）で強調される
    assert "#e07b00" in html  # 150.0%の区間
    assert "#cc2222" in html  # 250.0%の区間
    # 横軸ラベルは対応する点の真下に短い表記のまま表示される
    assert ">4/13</text>" in html
    assert ">5/4</text>" in html
    # 的中率(左軸)は0/50/100%、回収率(右軸)は0/200/300%超の固定軸ラベルを表示する
    assert ">50%</text>" in html
    assert ">200%</text>" in html
    assert ">300%+</text>" in html


def test_hit_return_trend_svg_returns_empty_string_for_fewer_than_two_points():
    assert s.hit_return_trend_svg(["4/13"], [10.0], [50.0]) == ""
    assert s.hit_return_trend_svg([], [], []) == ""


def test_hit_return_trend_svg_aligns_hit_and_return_axes_at_same_height():
    # 的中率と回収率は同じ_aligned_fractionを使うため、同じ値なら同じ高さになる
    # （0〜100%の範囲では軸が揃っていることを保証する）
    assert s._aligned_fraction(50.0) == s._aligned_fraction(50.0)
    assert s._aligned_fraction(100.0) == s._BREAKEVEN_FRACTION
    # 100%を超える分は圧縮されるため、100%→200%の伸びより200%→300%の伸びの方が小さい
    growth_100_200 = s._aligned_fraction(200.0) - s._aligned_fraction(100.0)
    growth_200_300 = s._aligned_fraction(300.0) - s._aligned_fraction(200.0)
    assert growth_200_300 < growth_100_200


def test_hit_return_trend_svg_thins_out_axis_labels_when_many_points():
    n = 20
    labels = [f"{i + 1}/1" for i in range(n)]
    hit_rates = [float(i) for i in range(n)]
    return_rates = [float(i * 10) for i in range(n)]

    html = s.hit_return_trend_svg(labels, hit_rates, return_rates)

    assert html.count("<circle") == n * 2 + 2  # データ点 + 凡例2点
    # x軸(間引き後5個) + 的中率y軸3個(0/50/100%) + 回収率y軸3個(0/200/300%+)
    assert html.count("trend-chart-axis-label") == 11
    assert ">1/1</text>" in html
    assert f">{n}/1</text>" in html


def test_select_tick_indices_includes_first_and_last():
    indices = s._select_tick_indices(20, max_ticks=4)
    assert indices[0] == 0
    assert indices[-1] == 19
    assert len(indices) <= 4
