# MAR 競馬AI - Claude Code ガイド

## プロジェクト概要

**MAR（まーる）** は競馬AIデータサイト（mar-keiba.com）のバックエンドシステム。
netkeiba をスクレイピングして出走データを収集し、LightGBM で予想スコアを生成、
HTML ページを生成して ConoHa VPS にアップロードする。

## 必須ルール（絶対に守ること）

### コミット・プッシュは明示指示があった時のみ
ファイル変更後は「変更完了」と報告するだけ。
「コミットして」「プッシュして」などの明示的な指示があって初めて実行する。

### ConoHa アップロードは自動実行しない
`upload_to_conoha.bat` / `upload_to_conoha_auto.bat` は絶対に自動で呼ばない。
ユーザーがまとめて手動実行する。git push 後は「アップロードはご自身でお願いします」と伝えるだけ。

## システム構成

### 主要ディレクトリ
- `src/` - Pythonソース（スクレイピング・予想・HTML生成）
- `bat/` - 自動化バッチファイル群
- `scripts/` - 手動実行スクリプト
- `data/` - レースデータ・モデルファイル（git 管理済み、33万ファイル）
- `public_html/` - 生成 HTML（ConoHa にアップロードする成果物）
- `logs/` - タスクスケジューラのログ

### 主要モジュール
- `src/logic/scraping/netkeiba_scraper.py` - netkeiba スクレイピング
- `src/logic/prediction/race_card_builder.py` - 予想スコア計算
- `src/logic/html_generator/race_page_generator.py` - 出馬表 HTML 生成
- `src/logic/html_generator/daily_index_generator.py` - 日別一覧 HTML
- `src/logic/html_generator/home_generator.py` - HOME 生成
- `src/logic/scheduler/race_day_scheduler.py` - 自動化スケジューラ
- `src/managers/race_card_dataset_manager.py` - データ読み書き
- `src/managers/race_schedule_dataset_manager.py` - 開催スケジュール管理

### タスクスケジューラ（16タスク、`\keiba\` フォルダ）
| タスク名 | 実行日時 | 内容 |
|---|---|---|
| weekly_update | 水 00:30 | 過去データ更新 |
| weekly_schedule | 木 20:00 | 翌週開催スケジュール・週末出馬表 HTML 作成 → 続けて週末展望コラム生成（Claude API） |
| make_html_prev_day | 金 18:00・土 16:00 | 翌日出馬表 HTML 生成 |
| post_weekend_preview | 金 12:00 | X (Twitter) 週末展望投稿 |
| generate_daily_preview | 金 20:00・土 20:00 | 当日AI予想コラム生成 |
| post_today_race | 土日 09:00 | 当日レース実況投稿（長時間ループ） |
| post_today_race_mon | 月 09:00 | 月曜レース実況投稿 |
| today_race_returns | 土日 18:00 | レース結果取得 |
| today_race_returns_mon | 月 18:00 | 月曜レース結果取得 |
| update_daily_html | 土日 20:00 | 当日 HTML 更新 |
| update_daily_html_mon | 月 20:00 | 月曜 HTML 更新 |
| generate_weekly_review | 火 21:00 | 週次振り返りコラム生成 |
| post_weekend_summary | 火 20:00 | 週末結果サマリー投稿 |
| update_today_content | 毎日 00:05 | HOME・カレンダー日次更新 |
| daily_trend | 土日 20:00 | 当日の傾向分析（開催なしの日は自動スキップ） |
| daily_trend_mon | 月 20:00 | 月曜の傾向分析 |

セットアップ: `bat\Deploy\setup_task_scheduler.bat` をダブルクリック（中で `setup_task_scheduler.ps1` を実行ポリシー回避付きで起動し、自動で管理者として起動し直す。最後は Enter を押すまで閉じない）。ユーザー名と python.exe のパスは自動で取得する。
- `.ps1` を右クリック →「PowerShell で実行」は使わない（実行ポリシーが Restricted のため、スクリプトがブロックされて画面がすぐ消える）
- このファイルは ASCII のみで書くこと（PowerShell 5.1 は BOM なしのファイルを Shift-JIS として読むため、日本語を入れると文法が崩れる）
- 前日オッズ更新は `generate_daily_preview` の最初の手順で行う。旧PCにあった `keiba_refresh_prev_day_odds` は二重実行になるので廃止した（セットアップ時に削除される）

**監視:** 全タスクは `scripts/run_task_with_notify.py` 経由で bat を実行する。開始時と完了時に Gmail（`.env` の `GMAIL_*`）で通知する。完了メールは ✅成功 / ⚠要確認（終了コードは 0 だがログにエラーらしき行がある）/ ❌失敗（終了コードが 0 以外）のいずれか。ログは `logs/tasks/<タスク名>_YYYYMMDD_HHMMSS.log` に保存し、60日で自動削除。テスト送信: `python scripts\run_task_with_notify.py --test`

**実行時間の上限:** タスクスケジューラの上限は基本2時間、`weekly_update` は6時間（このPCでは手順1だけで約1時間45分かかる）、`post_today_race(_mon)` は14時間。通知ラッパーには上限の10分前に `--timeout-min` で自分のタイムアウトを持たせており、超えたら bat をプロセスツリーごと止めて「❌タイムアウト」メールを送る（タスクスケジューラに上限で止められると完了メールが届かないため）。

## 予想モデル体制

### 本番モデル（3モデル体制）
- **MAR**: バランス型メインモデル
- **MAR-hit**: 的中率重視（`hit_` prefix）
- **MAR-val**: 回収率重視（`val_` prefix）

モデル関連コード: `src/managers/ai_performance_dataset_manager.py`（`MODEL_PREFIX`, `MODEL_LABELS`）

### モデル性能ベースライン（2026年全場1707R）
| モデル | 単勝的中率 | 単勝回収率 | 3連複的中率 | 3連複回収率 |
|---|---|---|---|---|
| v3blend（MAR-hit） | 39.4% | 193.0% | 41.5% | 169.6% |
| v4（MAR-val） | 34.0% | 298.5% | 31.8% | 379.0% |
| v5（血統Embedding） | 34.8% | 291.6% | 32.3% | 619.4% |

**場別推奨:**
- 小倉: v5（3連複3426%）
- 中山: v4（単勝461.2%）
- 新潟: v5（単勝318.9%）
- 福島: v3blend

### 研究モデルの評価まとめ
- **単勝回収率ベスト**: v7×v5prime α=0.9（97.8%）
- **3連複的中率ベスト**: v12×v7 α=0.4（28.4%）/ 3-way blend v7:0.4+v15:0.5+v18:0.1（28.8%）
- **3連複回収率ベスト**: v12×v7 α=0.7（106.4%）
- v10ev（EV回帰）は失敗。次の方向: Embedding+MLP（血統ID → 埋め込みベクトル）

## 環境設定

### 必要な環境変数（`.env` に記載、git 管理外）
- `ANTHROPIC_API_KEY` - コラム生成（Claude API）
- `NETKEIBA_LOGIN_ID` / `NETKEIBA_PASSWORD` - 調教タイム取得
- `GMAIL_ACCOUNT` / `GMAIL_APP_PASSWORD` / `GMAIL_SEND_TO` - メール送信
- `X_API_KEY` / `X_API_SECRET` / `X_ACCESS_TOKEN` / `X_ACCESS_TOKEN_SECRET` - X 投稿

### ConoHa アップロード
WinSCP にサイト名 `mar-keiba` として SFTP 設定を保存。
`bat\Deploy\upload_to_conoha.bat` で手動アップロード。

### Python パッケージ
```
pip install anthropic lightgbm numpy pandas requests tweepy beautifulsoup4 scipy tqdm python-dotenv Pillow jpholiday
```
- `jpholiday` は HTML 生成系（`src/utils/format_data.py`）が import するため必須。無いと15タスク中12タスクが起動直後に失敗する
- `torch` は研究用モデル（v13 MLP / v14 LSTM の学習・評価）のみで使用。本番の自動化には不要

## よくある作業パターン

### 特定レースの HTML を再生成
```python
# PYTHONIOENCODING=utf-8 が必要（✓ 文字の cp932 エラー対策）
PYTHONIOENCODING=utf-8 python -c "
import sys; sys.path.insert(0, '.')
from src.logic.scheduler import race_day_scheduler
from datetime import date
race_day_scheduler.make_html_prev_day(date(YYYY, MM, DD))
"
```

### 開催日の HTML 一括更新
```
bat\MakeHTML\make_html_prev_day.bat
```

### race_time_id_list の確認
`data/race_schedule/race_time_id_list/YYYYMMDD.csv`

## 注意事項

- `race_time` は HHMM 形式の文字列。括弧付きレース名（天皇賞(秋)など）は `_extract_race_time` のトークン位置がズレるバグが過去にあった（2026-11-01 修正済み）
- `make_html_prev_day` を直接 Python から呼ぶ際は `PYTHONIOENCODING=utf-8` が必要
- data/ は git 管理済み（33万ファイル）。clone は時間がかかる
