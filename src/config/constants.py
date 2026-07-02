# 開催場のid付きフォルダ名リスト（libs/name_header.py PLACE_LIST より）
PLACE_LIST = ["01_sapporo", "02_hakodate", "03_fukushima", "04_nigata", "05_tokyo",
              "06_nakayama", "07_chukyo", "08_kyoto", "09_hanshin", "10_kokura"]

# 開催場の名前リスト（libs/name_header.py NAME_LIST より）
NAME_LIST = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

# クラスのリスト（libs/name_header.py CLASS_LIST より）
CLASS_LIST = ["未勝利", "新馬", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"]

# 馬場状態のリスト（libs/name_header.py GROUND_STATE_LIST より）
GROUND_STATE_LIST = ["良", "稍重", "重", "不良"]

# 賭式のリスト（libs/name_header.py BETTING_TYPE_LIST より）
BETTING_TYPE_LIST = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"]

# 枠番の表示色（web/src/config/settings.py WAKU_COLORS より）
WAKU_COLORS = {
    "1": "white",   # 白
    "2": "black",   # 黒
    "3": "red",     # 赤
    "4": "blue",    # 青
    "5": "yellow",  # 黄
    "6": "green",   # 緑
    "7": "orange",  # 橙
    "8": "pink",    # 桃
}

# 着順の表示色（web/src/config/settings.py RANK_COLORS より）
RANK_COLORS = {
    "1": "#FFD700",  # 金色
    "2": "#B0E0E6",  # 水色
    "3": "#FFA07A",  # オレンジ
}

# LightGBMモデル学習時にGPUを使用するか
# True にすると LIGHTGBM_DEVICE が device= パラメータとして渡される
# GPU非搭載PCや環境未整備の場合は False のままにする
USE_GPU_TRAINING = False
# 使用するデバイス種別: 'gpu'（OpenCL）または 'cuda'（NVIDIA CUDA、より高速）
LIGHTGBM_DEVICE = "cuda"
