"""horse_peds（血統）データセットのManager層

CSVの読み込み・書き込み・永続化を担う。
変換ロジック自体は src/datasets/horse/transform.py に切り出している。

旧 src/legacy_datasets/horse_peds.py からの移植。
"""

import os

import numpy as np
from tqdm import tqdm

from src.config import paths
from src.datasets.horse import transform
from src.logic.scraping import netkeiba_scraper
from src.utils.file_utils import read_csv_or_empty

HORSE_PEDS_DATA_PATH = os.path.join(paths.HORSE_DATA_PATH, "horse_peds")


def get_horse_peds_csv(horse_id):
    """horse_idのhorse_pedsデータcsvを取得する

    Args:
        horse_id (str): horse_id

    Returns:
        pd.DataFrame: horse_pedsデータセット（存在しない場合は空のDataFrame）
    """
    path = os.path.join(HORSE_PEDS_DATA_PATH, f"{horse_id}.csv")
    df = read_csv_or_empty(path, dtype=str, index_col=0)
    if not df.empty:
        col = df.columns[0]
        df[col] = df[col].apply(transform.normalize_horse_name_strings)
    return df


def save_horse_peds_dataset(horse_id, peds_df):
    """horse_pedsのデータを保存する

    Args:
        horse_id (str): horse_id
        peds_df (pd.Series または pd.DataFrame): horse_pedsのデータ
    """
    if peds_df is None or peds_df.empty:
        return

    peds_df = peds_df.astype(str)
    os.makedirs(HORSE_PEDS_DATA_PATH, exist_ok=True)
    peds_df.to_csv(os.path.join(HORSE_PEDS_DATA_PATH, f"{horse_id}.csv"))
    peds_df.to_pickle(os.path.join(HORSE_PEDS_DATA_PATH, f"{horse_id}.pickle"))


def is_horse_peds_dataset(horse_id):
    """horse_idの血統データが既に存在するかを確認する

    Args:
        horse_id (str): horse_id

    Returns:
        bool: 既存ならTrue
    """
    path = os.path.join(HORSE_PEDS_DATA_PATH, f"{horse_id}.csv")
    return os.path.isfile(path)


def get_peds_info(horse_id):
    """血統情報(父・母父)を取得する。データが無い場合はスクレイピングして作成する。

    Args:
        horse_id (str): horse_id

    Returns:
        list: [父, 母父]
    """
    try:
        peds_data = get_horse_peds_csv(horse_id)
        if peds_data.empty:
            peds_data = netkeiba_scraper.scrape_horse_peds(horse_id)
            peds_data = transform.delete_invalid_strings(peds_data)
            save_horse_peds_dataset(horse_id, peds_data)
            peds_data = get_horse_peds_csv(horse_id)
        peds_list = peds_data[str(horse_id)].tolist()
        return [peds_list[0], peds_list[4]]
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
        return [np.nan, np.nan]


def make_horse_peds_datasets_from_horse_id_list(horse_id_list):
    """horse_id_listから未取得のhorse_pedsデータセットを作成する

    Args:
        horse_id_list (list): horse_idのリスト
    """
    try:
        for horse_id in tqdm(horse_id_list):
            if not is_horse_peds_dataset(horse_id):
                peds_df = netkeiba_scraper.scrape_horse_peds(horse_id)
                peds_df = transform.delete_invalid_strings(peds_df)
                save_horse_peds_dataset(horse_id, peds_df)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
