"""race_info（レース基本情報・集計）データセットの列定義"""

# 集計軸となるクラスのリスト（"all"=全クラス合算を含む）
CLASSES = ["all", "未勝利", "新馬", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"]

# 集計軸となる馬場状態のリスト（"全"=全馬場状態合算を含む）
GROUNDS = ["全", "良", "稍重", "重", "不良"]

# analyze_winner_weights / analyze_winner_weights_multi_years の出力列
WINNER_WEIGHT_COLUMNS = ["race_type", "course_len", "ground_state", "class", "馬体重"]

# analyze_average_pops / analyze_average_pop_multi_years の出力列
AVERAGE_POPS_COLUMNS = (
    ["race_type", "course_len", "ground_state", "class", "avg_pop"]
    + [f"pop_{i}_count" for i in range(1, 19)]
)

# analyze_average_frame_and_horse / analyze_frame_and_horse_multi_years の出力列
AVERAGE_FRAMES_COLUMNS = (
    ["race_type", "course_len", "ground_state", "class", "avg_frame", "avg_horse", "total_winners"]
    + [f"frame_{i}_wins" for i in range(1, 9)]
    + [f"horse_{j}_wins" for j in range(1, 19)]
)

# analyze_average_frame_and_horse_top3 / analyze_frame_and_horse_top3_multi_years の出力列
AVERAGE_FRAMES_TOP3_COLUMNS = (
    ["race_type", "course_len", "ground_state", "class", "avg_frame", "avg_horse", "total_top3"]
    + [f"frame_{i}_top3" for i in range(1, 9)]
    + [f"horse_{j}_top3" for j in range(1, 19)]
)

# horse_id_map.csv の列
HORSE_ID_MAP_COLUMNS = ["馬名", "horse_id"]
