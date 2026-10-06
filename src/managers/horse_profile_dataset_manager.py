"""horse_profile（馬プロフィール）データセットのManager層

生年月日などの馬固有情報をキャッシュ・管理する。
スクレイピングは src/logic/scraping/netkeiba_scraper.scrape_horse_profile() が担う。

保存先: data/horse/horse_profile/{horse_id}.csv
形式:   horse_id, birth_date (YYYY-MM-DD)
"""

import os
import time

import pandas as pd
from tqdm import tqdm

from src.config import paths
from src.logic.scraping import netkeiba_scraper
from src.utils.file_utils import read_csv_or_empty

HORSE_PROFILE_DATA_PATH = os.path.join(paths.HORSE_DATA_PATH, "horse_profile")


def _profile_path(horse_id):
    return os.path.join(HORSE_PROFILE_DATA_PATH, f"{horse_id}.csv")


def is_horse_profile(horse_id):
    """プロフィールキャッシュが存在するか確認する"""
    return os.path.isfile(_profile_path(str(horse_id)))


def get_horse_profile_csv(horse_id):
    """キャッシュCSVを読み込む。なければ空のDataFrameを返す"""
    return read_csv_or_empty(_profile_path(str(horse_id)), dtype=str, index_col=0)


def save_horse_profile(horse_id, df):
    """プロフィールデータをCSVに保存する"""
    if df is None or df.empty:
        return
    os.makedirs(HORSE_PROFILE_DATA_PATH, exist_ok=True)
    df.to_csv(_profile_path(str(horse_id)), index=False)


def get_horse_profile(horse_id):
    """プロフィールを取得する。キャッシュがなければスクレイピングして保存する。

    Args:
        horse_id (str): horse_id

    Returns:
        pd.DataFrame: 1行のDataFrame (horse_id, birth_date)。
                      取得失敗時は空のDataFrame。
    """
    df = get_horse_profile_csv(horse_id)
    if not df.empty:
        return df
    df = netkeiba_scraper.scrape_horse_profile(str(horse_id))
    if not df.empty:
        save_horse_profile(horse_id, df)
    return df


def get_birth_month(horse_id):
    """誕生月を返す（1〜12）。取得できない場合は None。

    キャッシュがあればそこから、なければスクレイピングして取得する。
    """
    try:
        df = get_horse_profile(str(horse_id))
        if df.empty:
            return None
        birth_date = str(df["birth_date"].iloc[0])
        month = int(birth_date.split("-")[1])
        return month if 1 <= month <= 12 else None
    except Exception:
        return None


def fetch_profiles_for_horse_ids(horse_id_list, skip_existing=True):
    """horse_idリストのプロフィールをまとめて取得・保存する（一括スクレイピング用）。

    Args:
        horse_id_list (list): horse_idのリスト
        skip_existing (bool): キャッシュ済みをスキップするか（デフォルト True）
    """
    targets = [
        hid for hid in horse_id_list
        if not (skip_existing and is_horse_profile(str(hid)))
    ]
    print(f"取得対象: {len(targets)}頭 / 全{len(horse_id_list)}頭")
    ok, ng = 0, 0
    consecutive_errors = 0
    for hid in tqdm(targets, desc="horse_profile取得"):
        df = netkeiba_scraper.scrape_horse_profile(str(hid))
        if not df.empty:
            save_horse_profile(hid, df)
            ok += 1
            consecutive_errors = 0
        else:
            ng += 1
            consecutive_errors += 1
            # 連続エラーが続く場合は長めに待機（レート制限対策）
            if consecutive_errors >= 10:
                time.sleep(30)
                consecutive_errors = 0
        time.sleep(2)
    print(f"完了: 成功{ok}頭 / 失敗{ng}頭")
