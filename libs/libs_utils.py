import os
import sys

# pycache を生成しない
sys.dont_write_bytecode = True
import name_header

def create_place_folders(base_path):
    """
    指定された base_path に PLACE_LIST のフォルダを作成する。

    Parameters
    ----------
    base_path : str
        フォルダを作成する親ディレクトリのパス
    """
    for place in name_header.PLACE_LIST:
        folder_path = os.path.join(base_path, place)
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"✅ 作成済み: {folder_path}")
        except Exception as e:
            print(f"❌ エラー: {folder_path} の作成に失敗しました → {e}")

if __name__ == '__main__':
    base_path = name_header.DATA_PATH + "//AverageFrames"
    create_place_folders(base_path)