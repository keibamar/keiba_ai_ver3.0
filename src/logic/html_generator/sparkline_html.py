"""数値の時系列トレンドをSVG折れ線（スパークライン）で表示するHTML部品

rate_gauge_html.py と同じ「外部ライブラリ不使用、文字列としてHTML/SVG断片を返す」
方針を踏襲する。年度別成績のような「値が並んでいるだけでは増減が分かりにくい」
内訳の直前に添えて、傾向を一目で把握できるようにする。
"""


def sparkline_svg(values, labels, width=180, height=40, color="#b04a5a"):
    """時系列の数値リストを折れ線（SVG polyline）として返す

    Args:
        values (list[float]): 古い→新しい順の数値リスト。
        labels (list[str]): valuesと同じ長さの、各点に対応するラベル
            （例: 年度）。最初と最後の値・ラベルのみ目盛りとして表示する。
        width (int): SVGの幅(px)。
        height (int): SVGの高さ(px)。
        color (str): 折れ線・点の色。

    Returns:
        str: データ点が2点未満の場合は空文字列（折れ線として描けないため）。
    """
    if len(values) < 2:
        return ""

    pad = 6
    min_v, max_v = min(values), max(values)
    span = max_v - min_v if max_v > min_v else 1.0
    n = len(values)

    def x_at(i):
        return pad + (width - 2 * pad) * i / (n - 1)

    def y_at(v):
        return height - pad - (height - 2 * pad) * (v - min_v) / span

    points = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(values))
    dots = "".join(
        f'<circle cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" r="2.5" fill="{color}"></circle>'
        for i, v in enumerate(values)
    )

    return f"""<span class="sparkline-wrap">
  <svg class="sparkline" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"></polyline>
    {dots}
  </svg>
  <span class="sparkline-labels">
    <span class="sparkline-label-start">{labels[0]}: {values[0]:.1f}</span>
    <span class="sparkline-label-end">{labels[-1]}: {values[-1]:.1f}</span>
  </span>
</span>"""
