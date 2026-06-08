import os
import sys

# pycache を生成しない
sys.dont_write_bytecode = True
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # web/src
sys.path.append(SRC_ROOT)

from config.path import HTML_HOME_PATH
from config.templates import load_template

TEMPLATE_NAME = "global_index.html"

# -----------------------------
# HTML生成
# -----------------------------
def generate_global_index():
    os.makedirs(HTML_HOME_PATH, exist_ok=True)

    template = load_template(TEMPLATE_NAME)
    html = template.render(
        title="MAR(まーる）|競馬AIデータサイト"
    )
    output_path = os.path.join(HTML_HOME_PATH, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("Generated: site/index.html")


# -----------------------------
# 実行
# -----------------------------
if __name__ == "__main__":
    generate_global_index()
