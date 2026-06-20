"""的中率・回収率を色付きゲージバーで表示するHTML部品

ai_performance_report_generator（成績テーブル）とhome_generator（週別推移グラフ）の
両方から共通で使われる。数値だけでは良し悪しが分かりにくいため、的中率は
低い（青）→高い（赤）のグラデーションで、回収率は損益分岐点である100%を
ゲージ上に基準線として表示し、100%超は橙色、200%超は赤色で塗り分ける。
"""

HIT_RATE_GRADIENT_MAX = 50.0  # この値以上の的中率は最大の赤色として扱う
RETURN_RATE_GAUGE_MAX = 200.0  # ゲージの右端が表す回収率（100%が中央に来る）

_HIT_RATE_COLOR_LOW = (31, 79, 214)  # 青
_HIT_RATE_COLOR_HIGH = (214, 47, 47)  # 赤


def _hit_rate_color(hit_rate):
    t = max(min(hit_rate / HIT_RATE_GRADIENT_MAX, 1.0), 0.0)
    r = round(_HIT_RATE_COLOR_LOW[0] + (_HIT_RATE_COLOR_HIGH[0] - _HIT_RATE_COLOR_LOW[0]) * t)
    g = round(_HIT_RATE_COLOR_LOW[1] + (_HIT_RATE_COLOR_HIGH[1] - _HIT_RATE_COLOR_LOW[1]) * t)
    b = round(_HIT_RATE_COLOR_LOW[2] + (_HIT_RATE_COLOR_HIGH[2] - _HIT_RATE_COLOR_LOW[2]) * t)
    return f"rgb({r},{g},{b})"


def _return_rate_color(return_rate):
    if return_rate > 200.0:
        return "#cc2222"
    if return_rate > 100.0:
        return "#e07b00"
    return "#1f4fd6"


def hit_rate_gauge_html(hit_rate):
    """的中率（0〜100%）を、値が高いほど赤くなるゲージバーHTMLとして返す"""
    width = max(min(hit_rate, 100.0), 0.0)
    color = _hit_rate_color(hit_rate)
    return (
        '<span class="rate-gauge">'
        f'<span class="gauge-track"><span class="gauge-fill" style="width: {width:.1f}%; background: {color};"></span></span>'
        f'<span class="gauge-value">{hit_rate:.1f}%</span>'
        "</span>"
    )


def return_rate_gauge_html(return_rate):
    """回収率を、100%基準線付きのゲージバーHTMLとして返す

    ゲージの右端を200%として、100%の位置（損益分岐点）に基準線を表示する。
    100%以下は青、100%超は橙、200%超は赤で塗り分ける（200%超もゲージ幅は満タンで止める）。
    """
    width = max(min(return_rate, RETURN_RATE_GAUGE_MAX), 0.0) / RETURN_RATE_GAUGE_MAX * 100.0
    color = _return_rate_color(return_rate)
    return (
        '<span class="rate-gauge">'
        '<span class="gauge-track">'
        f'<span class="gauge-fill" style="width: {width:.1f}%; background: {color};"></span>'
        '<span class="gauge-marker"></span>'
        "</span>"
        f'<span class="gauge-value">{return_rate:.1f}%</span>'
        "</span>"
    )
