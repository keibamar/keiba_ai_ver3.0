"""src/logic/html_generator/race_type_badge_html.py のテスト（オフライン）。"""

from src.logic.html_generator import race_type_badge_html as b


def test_race_type_span_html_colors_turf():
    assert b.race_type_span_html("芝") == '<span class="race-type-turf">芝</span>'


def test_race_type_span_html_colors_dirt():
    assert b.race_type_span_html("ダート") == '<span class="race-type-dirt">ダート</span>'


def test_race_type_span_html_returns_plain_text_for_unknown_value():
    assert b.race_type_span_html("障害") == "障害"


def test_course_label_html_wraps_the_whole_label_including_distance():
    assert b.course_label_html("芝", 1400) == '<span class="race-type-turf">芝1400m</span>'
    assert b.course_label_html("ダート", 1600) == '<span class="race-type-dirt">ダート1600m</span>'


def test_grade_badge_html_renders_g1_g2_g3():
    assert b.grade_badge_html("G1") == '<span class="grade-badge grade-badge-G1">G1</span>'
    assert b.grade_badge_html("G2") == '<span class="grade-badge grade-badge-G2">G2</span>'
    assert b.grade_badge_html("G3") == '<span class="grade-badge grade-badge-G3">G3</span>'


def test_grade_badge_html_returns_empty_for_non_graded_or_none():
    assert b.grade_badge_html(None) == ""
    assert b.grade_badge_html("Listed") == ""
