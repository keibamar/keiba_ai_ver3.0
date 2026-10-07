"""調教タイムデータを取得してCSVに保存するスクリプト

使い方:
    python scripts/fetch_training_data.py --login_id YOUR_ID --password YOUR_PW --horse_ids 2019102632 2020100001

引数:
    --login_id   netkeibaログインID（メールアドレス）
    --password   netkeibaパスワード
    --horse_ids  取得する horse_id（複数指定可）
    --output_dir 出力ディレクトリ（デフォルト: data/training）
    --sleep      リクエスト間隔秒数（デフォルト: 1.5）

出力:
    data/training/{horse_id}.csv  — 馬ごとの調教タイムCSV
    data/training/all_training.csv — 全馬まとめCSV（--horse_ids 複数のとき）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logic.scraping.training_scraper import (
    login_netkeiba,
    scrape_horse_training,
    scrape_multiple_horses,
)


def main():
    parser = argparse.ArgumentParser(description="netkeiba調教タイムスクレイパー")
    parser.add_argument("--login_id", required=True, help="netkeibaログインID")
    parser.add_argument("--password", required=True, help="netkeibaパスワード")
    parser.add_argument(
        "--horse_ids", nargs="+", required=True, help="取得するhorse_id（複数可）"
    )
    parser.add_argument(
        "--output_dir", default="data/training", help="出力ディレクトリ"
    )
    parser.add_argument(
        "--sleep", type=float, default=1.5, help="リクエスト間隔秒数"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("netkeibaにログイン中...")
    session = login_netkeiba(args.login_id, args.password)
    print("ログイン完了")

    if len(args.horse_ids) == 1:
        horse_id = args.horse_ids[0]
        print(f"horse_id={horse_id} の調教データを取得中...")
        df = scrape_horse_training(horse_id, session)
        if df.empty:
            print("データが取得できませんでした")
            return
        out_path = output_dir / f"{horse_id}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"保存完了: {out_path}  ({len(df)} 件)")
    else:
        df = scrape_multiple_horses(args.horse_ids, session, sleep_sec=args.sleep)
        if df.empty:
            print("データが取得できませんでした")
            return

        # 馬ごとにも保存
        for horse_id, group in df.groupby("horse_id"):
            out_path = output_dir / f"{horse_id}.csv"
            group.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  {horse_id}: {len(group)} 件 → {out_path}")

        # 全馬まとめ
        all_path = output_dir / "all_training.csv"
        df.to_csv(all_path, index=False, encoding="utf-8-sig")
        print(f"\n全馬まとめ保存: {all_path}  ({len(df)} 件)")


if __name__ == "__main__":
    main()
