import os

import pandas as pd

from src.config.constants import PLACE_LIST


def create_place_folders(base_path):
    """base_path配下にPLACE_LISTの開催場フォルダを作成する

    Args:
        base_path (str): フォルダを作成する親ディレクトリのパス
    """
    for place in PLACE_LIST:
        os.makedirs(os.path.join(base_path, place), exist_ok=True)


def read_csv_or_empty(path, dtype=str, index_col=None):
    """CSVが存在すれば読み込み、存在しなければ空のDataFrameを返す

    Args:
        path (str): CSVファイルパス
        dtype: pandas.read_csvに渡すdtype指定
        index_col: pandas.read_csvに渡すindex_col指定

    Returns:
        pd.DataFrame: 読み込んだDataFrame、存在しない場合は空のDataFrame
    """
    if os.path.isfile(path):
        return pd.read_csv(path, dtype=dtype, index_col=index_col)
    return pd.DataFrame()
