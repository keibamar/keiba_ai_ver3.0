import os
import re
from glob import glob


def get_subfolders(path):
    """
    指定したフォルダ内にあるフォルダ名一覧を取得する関数
    :param path: 親フォルダのパス
    :return: サブフォルダ名のリスト
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"指定されたパスが存在しません: {path}")
    
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def parse_filename(filename: str):
    """
    ファイル名から各変数を抽出する。
    形式: YYYYPPKKDDRR.csv
      YYYY = 年（4桁）
      PP   = 開催場ID（2桁）
      KK   = 開催数（2桁）
      DD   = 開催日（2桁）
      RR   = レース番号（2桁）
    """
    # print(filename)
    base = os.path.splitext(os.path.basename(filename))[0]  # 拡張子を除いた部分
    if not re.match(r"^\d{12}$", base):
        raise ValueError(f"ファイル名の形式が不正です: {filename}")

    year = base[0:4]
    place_id = base[4:6]
    times = base[6:8]
    days = base[8:10]
    race_num = base[10:12]

    return {
        "year": int(year),
        "place_id": int(place_id),
        "times": int(times),
        "days": int(days),
        "race_num": int(race_num)
    }

def list_files_and_parse(input_dir : str):
    """
    指定ディレクトリ内のCSVファイルを走査し、
    拡張子を除いたファイル名と分解した変数を返す。
    """
    files = glob(os.path.join(input_dir, "*.csv"))
    results = []
    for f in files:
        try:
            parsed = parse_filename(f)
            results.append({"file": os.path.splitext(os.path.basename(f))[0], **parsed})
        except ValueError as e:
            print(e)
    return results