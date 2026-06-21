"""的中率（参考・左軸）・回収率（主役・右軸）の推移を1つの折れ線グラフで表示するHTML部品

rate_gauge_html.py と同じ「外部ライブラリ不使用、文字列としてHTML/SVG断片を返す」
方針を踏襲する。的中率・回収率を別々の小さなグラフに分けると見比べにくく、また
単純な折れ線だけでは「何を表しているか分かりにくい」ため、以下を加えて読み取りやすくする:
  - 的中率（左軸）と回収率（右軸）を同じグラフに重ねて表示する
  - 両軸は0〜100%の範囲で同じ高さになるよう揃える（的中率は100%が上限のため、
    この範囲だけで全体を表せる）。回収率が100%を超える分は、軸の残り部分に
    平方根で圧縮したおおよその比率で詰め込み、極端に高い回収率でもグラフが
    縦に間延びしないようにする
  - 的中率はあくまで参考データという位置づけのため、グレーの破線で薄く表示する
  - 回収率は損益分岐点である100%に点線の基準線を引き、100%を超える点・区間は
    return_rate_color（青→橙→赤）と同じ基準で色を変えて強調する
  - 各点にネイティブのtitle属性のツールチップを付け、ホバーで日付・値を確認できる
  - 横軸ラベルはグラフの真下に対応する点ごとに配置し、点数が多い場合は間引く
"""

from src.logic.html_generator.rate_gauge_html import return_rate_color

_MAX_AXIS_TICKS = 5
_HIT_RATE_COLOR = "#aaaaaa"
_BREAKEVEN_FRACTION = 0.65  # 0〜100%が占める高さの割合（残りを100%超の圧縮表示に使う）
_COMPRESSED_MAX = 300.0  # 圧縮表示の上限とみなす回収率（これ以上は頭打ちで表示する）


def _select_tick_indices(n, max_ticks):
    """0〜n-1のうち、最初・最後を含めておおむね等間隔なmax_ticks個のインデックスを選ぶ"""
    if n <= max_ticks:
        return list(range(n))
    step = (n - 1) / (max_ticks - 1)
    return sorted({round(i * step) for i in range(max_ticks)})


def _aligned_fraction(value):
    """的中率・回収率共通の「高さの割合」（0.0〜1.0）を返す

    0〜100%は一定の比率（_BREAKEVEN_FRACTIONまで線形）で、的中率と回収率の軸を
    揃える。的中率は100%が上限のためこの範囲だけで表現できるが、回収率は
    100%を超えることがあるため、その分は残りの高さにおおよその比率（平方根）で
    圧縮して詰め込む。
    """
    value = max(value, 0.0)
    if value <= 100.0:
        return value / 100.0 * _BREAKEVEN_FRACTION
    over = min(value - 100.0, _COMPRESSED_MAX - 100.0)
    extra = (over / (_COMPRESSED_MAX - 100.0)) ** 0.5
    return _BREAKEVEN_FRACTION + extra * (1.0 - _BREAKEVEN_FRACTION)


def hit_return_trend_svg(labels, hit_rates, return_rates, width=320, height=170):
    """的中率（左軸・グレーの参考データ）・回収率（右軸・損益分岐点で色分け）の
    推移を、軸を揃えた1つの折れ線グラフで返す

    Args:
        labels (list): 古い→新しい順の各点のラベル（短い表記を推奨。例: "4/13", 2024）。
        hit_rates (list[float]): labelsと同じ長さの的中率（%）。
        return_rates (list[float]): labelsと同じ長さの回収率（%）。

    Returns:
        str: データ点が2点未満の場合は空文字列。
    """
    if len(labels) < 2:
        return ""

    labels = [str(label) for label in labels]
    n = len(labels)

    pad_left, pad_right, pad_top, pad_bottom = 30, 36, 22, 16
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    baseline_y = pad_top + plot_h

    def x_at(i):
        return pad_left + plot_w * i / (n - 1)

    def y_at(v):
        return baseline_y - plot_h * _aligned_fraction(v)

    hit_points = [(x_at(i), y_at(v)) for i, v in enumerate(hit_rates)]
    return_points = [(x_at(i), y_at(v)) for i, v in enumerate(return_rates)]

    hit_line = " ".join(f"{x:.1f},{y:.1f}" for x, y in hit_points)
    hit_dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{_HIT_RATE_COLOR}">'
        f"<title>{labels[i]} 的中率: {hit_rates[i]:.1f}%</title></circle>"
        for i, (x, y) in enumerate(hit_points)
    )

    return_segments = "".join(
        f'<line x1="{return_points[i][0]:.1f}" y1="{return_points[i][1]:.1f}" '
        f'x2="{return_points[i + 1][0]:.1f}" y2="{return_points[i + 1][1]:.1f}" '
        f'stroke="{return_rate_color(max(return_rates[i], return_rates[i + 1]))}" stroke-width="2.5"></line>'
        for i in range(n - 1)
    )
    return_dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" fill="{return_rate_color(return_rates[i])}">'
        f"<title>{labels[i]} 回収率: {return_rates[i]:.1f}%</title></circle>"
        for i, (x, y) in enumerate(return_points)
    )

    breakeven_y = y_at(100.0)
    breakeven = (
        f'<line x1="{pad_left}" y1="{breakeven_y:.1f}" x2="{width - pad_right}" y2="{breakeven_y:.1f}" '
        'class="trend-chart-breakeven"></line>'
        f'<text x="{width - pad_right + 3}" y="{breakeven_y + 3:.1f}" class="trend-chart-breakeven-label">100%</text>'
    )

    tick_indices = _select_tick_indices(n, _MAX_AXIS_TICKS)
    axis_y = height - 3
    x_axis_labels = "".join(
        f'<text x="{x_at(i):.1f}" y="{axis_y}" class="trend-chart-axis-label" text-anchor="middle">{labels[i]}</text>'
        for i in tick_indices
    )

    # 的中率（左軸）は0/50/100%、回収率（右軸）は0/100/200/300%超を、共通のスケールで配置する
    hit_axis_labels = "".join(
        f'<text x="2" y="{y_at(v) + 3:.1f}" class="trend-chart-axis-label trend-chart-axis-hit">{v:.0f}%</text>'
        for v in (0.0, 50.0, 100.0)
    )
    return_axis_labels = (
        f'<text x="{width - 2}" y="{y_at(0.0) + 3:.1f}" class="trend-chart-axis-label trend-chart-axis-return" text-anchor="end">0%</text>'
        f'<text x="{width - 2}" y="{y_at(200.0) + 3:.1f}" class="trend-chart-axis-label trend-chart-axis-return" text-anchor="end">200%</text>'
        f'<text x="{width - 2}" y="{y_at(_COMPRESSED_MAX) + 3:.1f}" class="trend-chart-axis-label trend-chart-axis-return" text-anchor="end">{_COMPRESSED_MAX:.0f}%+</text>'
    )

    legend = (
        '<g class="trend-chart-legend">'
        f'<circle cx="{pad_left}" cy="10" r="4" fill="{_HIT_RATE_COLOR}"></circle>'
        f'<text x="{pad_left + 8}" y="13" class="trend-chart-legend-label">的中率(左軸・参考)</text>'
        f'<circle cx="{pad_left + 86}" cy="10" r="4" fill="#e07b00"></circle>'
        f'<text x="{pad_left + 94}" y="13" class="trend-chart-legend-label">回収率(右軸・100%超で橙/赤)</text>'
        "</g>"
    )

    return f"""<span class="trend-chart-wrap">
  <svg class="trend-chart" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    {legend}
    {breakeven}
    {hit_axis_labels}
    {return_axis_labels}
    <polyline points="{hit_line}" fill="none" stroke="{_HIT_RATE_COLOR}" stroke-width="1.5" stroke-dasharray="3 2" class="trend-chart-hit-line"></polyline>
    {return_segments}
    {hit_dots}
    {return_dots}
    {x_axis_labels}
  </svg>
</span>"""
