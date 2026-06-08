import os
import sys
from datetime import datetime, date

# 競馬場ごとのコース情報のリスト
COURSE_LISTS =[ [["芝","1000"],["芝","1200"],["芝","1500"],["芝","1800"],["芝","2000"],["芝","2600"],
                 ["ダート","1000"],["ダート","1700"],["ダート","2400"]],                                                      # 01_sapporo
                [["芝","1000"],["芝","1200"],["芝","1800"],["芝","2000"],["芝","2600"],
                 ["ダート","1000"],["ダート","1700"],["ダート","2400"]],                                                      # 02_hakodate
                [["芝","1200"],["芝","1800"],["芝","2000"],["芝","2600"],
                 ["ダート","1150"],["ダート","1700"],["ダート","2400"]],                                                      # 03_fukushima
                [["芝","1000"],["芝","1200"],["芝","1400"],["芝","1600"],["芝","1800"],["芝","2000"],["芝","2200"],["芝","2400"],
                 ["ダート","1200"],["ダート","1800"],["ダート","2500"]],                                                      # 04_nigata
                [["芝","1400"],["芝","1600"],["芝","1800"],["芝","2000"],["芝","2300"],["芝","2400"],["芝","2500"],["芝","3400"],
                 ["ダート","1300"],["ダート","1400"],["ダート","1600"],["ダート","2100"],["ダート","2400"]],                   # 05_tokyo
                [["芝","1200"],["芝","1600"],["芝","1800"],["芝","2000"],["芝","2200"],["芝","2500"],["芝","3600"],
                 ["ダート","1200"],["ダート","1800"],["ダート","2400"],["ダート","2500"]],                                     # 06_nakayama
                [["芝","1200"],["芝","1400"],["芝","1600"],["芝","2000"],["芝","2200"],
                 ["ダート","1200"],["ダート","1400"],["ダート","1800"],["ダート","1900"]],                                     # 07_chukyo
                [["芝","1200"],["芝","1400"],["芝","1600"],["芝","1800"],["芝","2000"],["芝","2200"],["芝","2400"],["芝","3000"],["芝","3200"],
                 ["ダート","1200"],["ダート","1400"],["ダート","1800"],["ダート","1900"]],                                     # 08_kyoto
                [["芝","1200"],["芝","1400"],["芝","1600"],["芝","1800"],["芝","2000"],["芝","2200"],["芝","2400"],["芝","2600"],["芝","3000"],
                 ["ダート","1200"],["ダート","1400"],["ダート","1800"],["ダート","2000"]],                                     # 09_hanshin
                [["芝","1200"],["芝","1700"],["芝","1800"],["芝","2000"],["芝","2600"],
                 ["ダート","1000"],["ダート","1700"],["ダート","2400"]],                                                       # 10_kokura          
              ]

# pycache を生成しない
sys.dont_write_bytecode = True
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # web/src
sys.path.append(SRC_ROOT)

from config.path import TRACK_MAP, PERFORMACE_HTML_PATH
from config.templates import load_template

TRACK_TEMPLATE_NAME = "track.html"
COURSE_TEMPLATE_NAME = "course.html"
INDEX_TEMPLATE_NAME = "course_index.html"

# -----------------------------
# HTML生成関数
# -----------------------------
def generate_track_pages():
    for idx, track_name in enumerate(TRACK_MAP.values()):
        track_dir = os.path.join(PERFORMACE_HTML_PATH, "course", track_name)
        os.makedirs(track_dir, exist_ok=True)

        course_list = COURSE_LISTS[idx]
        track_template = load_template(TRACK_TEMPLATE_NAME)
        course_template = load_template(COURSE_TEMPLATE_NAME)

        # -----------------------------
        # 競馬場全体ページ（index.html）
        # -----------------------------
        track_html = track_template.render(
            track_name=track_name,
            courses=course_list
        )

        with open(os.path.join(track_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(track_html)

        # -----------------------------
        # 各コースページ
        # -----------------------------
        for course in course_list:
            surface = course[0]   # 芝 or ダート
            distance = course[1]  # 1000, 1200, ...

            filename = f"{surface}-{distance}.html".replace("芝", "turf").replace("ダート", "dirt")
            
            course_html = course_template.render(
                track_name=track_name,
                surface=surface,
                distance=distance
            )

            with open(os.path.join(track_dir, filename), "w", encoding="utf-8") as f:
                f.write(course_html)

        print(f"Generated: {track_name}")

# -----------------------------
# 競馬場一覧ページを生成
# -----------------------------
def generate_course_index_page():

    # テンプレート読み込み
    course_list_template = load_template(INDEX_TEMPLATE_NAME)

    html = course_list_template.render(
        title="競馬場別成績",
        tracks=TRACK_MAP.values()
    )

    with open(os.path.join(PERFORMACE_HTML_PATH, "course", "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("Generated: ai/tracks/index.html")

# -----------------------------
# 実行
# -----------------------------
if __name__ == "__main__":
    generate_track_pages()
    generate_course_index_page()

