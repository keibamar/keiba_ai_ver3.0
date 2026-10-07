"""調教タイムデータのManager層

data/training/{horse_id}.csv を管理する。
ログイン認証は .env の NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD から取得する。

主なAPI:
    ensure_training_data(horse_id)  — データが古いor無ければ自動取得
    get_training_data(horse_id)     — CSV から DataFrame を返す
"""

import os
import time

import pandas as pd
import requests

from src.config import paths
from src.utils.file_utils import read_csv_or_empty

# 何日以内のデータなら再取得しない（レースサイクルに合わせた閾値）
_FRESHNESS_DAYS = 14

_session: requests.Session | None = None


def _get_session() -> requests.Session | None:
    """認証済みセッションを返す（遅延初期化・シングルトン）。

    .env の NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD が未設定なら None を返す。
    """
    global _session
    if _session is not None:
        return _session

    login_id = os.environ.get("NETKEIBA_LOGIN_ID", "")
    password = os.environ.get("NETKEIBA_PASSWORD", "")
    if not login_id or not password:
        return None

    from src.logic.scraping.training_scraper import login_netkeiba
    try:
        _session = login_netkeiba(login_id, password)
        print("  [調教] netkeibaログイン完了")
    except Exception as e:
        print(f"  [調教] ログイン失敗: {e}")
        _session = None
    return _session


def _csv_path(horse_id: str) -> str:
    return os.path.join(paths.TRAINING_DATA_PATH, f"{horse_id}.csv")


def _is_fresh(horse_id: str) -> bool:
    """CSVが存在し、最新の調教日が _FRESHNESS_DAYS 日以内なら True。"""
    path = _csv_path(horse_id)
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, usecols=["training_date"], parse_dates=["training_date"])
        if df.empty:
            return False
        latest = df["training_date"].max()
        age_days = (pd.Timestamp.now() - latest).days
        return age_days <= _FRESHNESS_DAYS
    except Exception:
        return False


def ensure_training_data(horse_id: str, sleep_sec: float = 1.0) -> None:
    """データが存在しないか古ければネットから取得して保存する。

    ログイン情報が .env に設定されていない場合は何もしない（エラーにしない）。
    """
    if _is_fresh(horse_id):
        return

    session = _get_session()
    if session is None:
        return

    from src.logic.scraping.training_scraper import scrape_horse_training
    try:
        print(f"  [調教] {horse_id} を取得中...")
        df = scrape_horse_training(horse_id, session)
        if not df.empty:
            os.makedirs(paths.TRAINING_DATA_PATH, exist_ok=True)
            df.to_csv(_csv_path(horse_id), index=False, encoding="utf-8-sig")
            print(f"  [調教] {horse_id}: {len(df)} 件保存")
        time.sleep(sleep_sec)
    except Exception as e:
        print(f"  [調教] {horse_id} 取得失敗: {e}")


def get_training_data(horse_id: str) -> pd.DataFrame:
    """調教タイムデータを DataFrame で返す（存在しなければ空 DataFrame）。"""
    path = _csv_path(horse_id)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def reset_session() -> None:
    """ログインセッションをリセットする（テスト用・再ログインが必要な場合）。"""
    global _session
    _session = None
