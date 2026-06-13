"""horse（血統・過去成績）データセットの列定義"""

# past_performance（過去成績）データセットの列順
PAST_PERFORMANCE_COLUMNS = [
    "race_id", "日付", "開催", "天気", "R", "レース名", "class", "頭数",
    "枠番", "馬番", "オッズ", "人気", "着順", "騎手", "斤量",
    "race_type", "course_len", "ground_state", "タイム", "着差",
    "通過", "上り", "馬体重", "勝ち馬 (2着馬)",
]

# aggregate_total_peds_results でのクラス表示順
CLASS_ORDER = ["all", "未勝利", "新馬", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"]

# 開催地名 -> place_id（2桁文字列）。normalize_past_performance_format でのrace_id生成に使用
PLACE_MAP = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06",
    "中京": "07", "京都": "08", "阪神": "09", "小倉": "10",
}
