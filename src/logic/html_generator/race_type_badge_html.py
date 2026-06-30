"""芝/ダートの色分け表示（Forge: HTMLFactory）のHTML断片

コース一覧・右側タブの階層表示など、芝/ダートのラベルが並ぶ箇所で一目で区別
できるよう、race_type（"芝"/"ダート"）に対応する色分けクラス付きのspanで返す。
"""

RACE_TYPE_CLASSES = {
    "芝": "race-type-turf",
    "ダート": "race-type-dirt",
}


def race_type_css_class(race_type):
    """race_typeに対応する色分けCSSクラス名を返す（該当が無ければ空文字）"""
    return RACE_TYPE_CLASSES.get(race_type, "")


def race_type_span_html(race_type):
    """race_typeの文字をそのまま、対応する色分けクラス付きのspanで返す

    芝/ダート以外（該当クラスが無い値）はそのままのテキストを返す。
    """
    css_class = race_type_css_class(race_type)
    if not css_class:
        return race_type
    return f'<span class="{css_class}">{race_type}</span>'


def course_label_html(race_type, course_len):
    """「芝1400m」のような表示ラベルを返す

    芝/ダートの文字だけでなく、続く距離（例: "1400m"）も含めたラベル全体を
    同じ色のspanで囲み、一覧で並んだときに芝/ダートを一目で区別しやすくする。
    """
    css_class = race_type_css_class(race_type)
    label = f"{race_type}{course_len}m"
    if not css_class:
        return label
    return f'<span class="{css_class}">{label}</span>'


def grade_badge_html(grade):
    """G1/G2/G3の重賞バッジ（色分けされた小さなラベル）を返す

    開催日カレンダー・開催一覧・HOME等、レース名を表示する複数の場所で共通して
    使う（.calendar-day-meeting-main内のJS版バッジ用CSSクラス名と揃えている）。
    対象外（Noneや未対応の値）の場合は空文字列を返す。
    """
    if grade not in ("G1", "G2", "G3"):
        return ""
    return f'<span class="grade-badge grade-badge-{grade}">{grade}</span>'
