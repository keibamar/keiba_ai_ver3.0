import os
from jinja2 import Environment, FileSystemLoader

# テンプレートディレクトリはこのファイル(`config`)と同階層の `../templates` ではなく
# `web/src/templates` を指す絶対パスに解決する
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# -----------------------------
# Jinja2 環境
# -----------------------------
# HTML を直接生成する用途が多いため自動エスケープは False にしている既存挙動を維持
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)


def load_template(template_name):
    return env.get_template(template_name)