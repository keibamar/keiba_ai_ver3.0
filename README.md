# keiba_ai_ver3.0

netkeiba.com / JRA公式サイトからレースデータ・血統・配当などをスクレイピングして
データセットを蓄積し、出走馬の予想を行う競馬予想AIプロジェクト。

`specifications/新設計.md` で定義した新アーキテクチャ（`src/{datasets,managers,logic,
output,config,utils}` の6層構造 × 7モジュール）への移行を、ドメインごとに
段階的に進めている。

## 1. リファクタリングの進捗状況（2026-06-14時点）

**全体は未完了。** データ収集・蓄積系（Chronicle / Atlas / Reaper 相当）、
予想エンジン（Oracle）、HTML生成系（Forge）、配信系（Herald・予想テキスト生成＋配信のみ）は
新構造への移行が完了しているが、配当結果レポート（Herald残部）と旧ファイルの
クリーンアップは未着手。

### 完了済み

| ドメイン | 旧実装 | 新実装 |
|---|---|---|
| race_schedule（Chronicle） | `libs/get_race_id.py` | `src/datasets/race_schedule/`, `src/managers/race_schedule_dataset_manager.py`, `src/logic/scraping/jra_calendar_scraper.py` |
| race_result（Atlas/Reaper） | `src/legacy_datasets/race_results.py` | `src/datasets/race_result/`, `src/managers/race_result_dataset_manager.py`, `src/logic/scraping/netkeiba_scraper.py`, `src/logic/scheduler/race_result_scheduler.py` |
| horse（Atlas） | `horse_peds.py` / `peds_results.py` / `past_performance.py` | `src/datasets/horse/`, `src/managers/{horse_peds,peds_results,past_performance}_dataset_manager.py`, `src/logic/scheduler/horse_scheduler.py` |
| race_info / race_returns（Atlas） | `analysis_race_info.py` / `analysis_race_time.py` / `average_time.py` / `race_returns.py` | `src/datasets/race_info/`, `src/managers/race_info_dataset_manager.py`, `src/logic/scheduler/{race_info_scheduler,race_returns_scheduler}.py` |
| config / utils | `libs/name_header.py` 等 | `src/config/{paths,constants,lists}.py`, `src/utils/file_utils.py` |
| 認証情報 | コード内ハードコード | `.env`（`libs/mail_api.py`, `libs/post_text.py` が読み込み） |
| 予想エンジン（Oracle・日次予想パスのみ） | `src/PredictionModels/LightGBM/{make_dataset,prediction}.py`, `src/RacePrediction/day_race_prediction.py` | `src/logic/prediction/race_prediction_engine.py`（`day_race_prediction.py` はこれへの薄いリダイレクトとして残置） |
| race_card / HTML生成（Forge） | `src/RacePrediction/race_card.py` / `make_time_id_list.py`（出力先のみ）, `web/src/generators/{race_pages,horse_info,daily_index,make_race_card_html}.py` | `src/managers/{race_card_dataset_manager,html_manager}.py`, `src/logic/html_generator/{race_page_generator,horse_report_generator,daily_index_generator}.py`, `src/utils/format_data.py`, `public_html/` |
| 予想テキスト生成・配信（Herald） | `src/RacePrediction/make_text.py`（`extract_top5_pred`/`make_race_text`）, `libs/{mail_api,post_text}.py` | `src/output/prediction_publisher.py` |

### 未対応（今後のフェーズ）

- **average_calculator**: 集計ロジックの一部 → `src/logic/calculators/average_calculator.py`
- **Oracle オフライン学習パイプライン**: `src/PredictionModels/LightGBM/`の`make_dataset_for_train`,
  `lightGBM_rank_train`, `prediction_rank`, `weekly_update_dataset_for_train`,
  `make_annual_dataset` 等は対象外（旧実装のまま、ver2.0データパスを参照し続ける）
- **Herald残部（配当結果レポート・日次配信オーケストレーション）**: `make_text.make_return_text`/
  `write_{win,place,trio}_hit_text`、`src/RacePrediction/calc_returns.py`（配当結果の的中率・
  回収率計算。ver2.0の`race_returns`モジュール・旧`DATA_PATH/RaceReturns`に依存し移植範囲が
  大きい）、`src/RacePrediction/post_daily_race.py`（日次配信オーケストレーションループ）は対象外
- **クリーンアップ**: `libs/`, `src/legacy_datasets/`, `src/RacePrediction/`, `web/`等の削除（呼び出し元を新実装に切り替えた後）

→ **データ収集・蓄積（週次/月次/年次更新）、予想エンジンの日次予想パス（Oracle）、
レースページ・日次インデックスのHTML生成（Forge）、予想テキスト生成・メール/X配信
（Herald・`src/output/prediction_publisher.py`）は新実装で動かせる**。
**配当結果レポート・日次配信オーケストレーション（`bat/TodayRace/post_today_race.bat`が
呼ぶ`post_daily_race.py`等）は、現状旧実装のままで、このリファクタリングによる影響はない。**

## 2. ディレクトリ構成（新実装部分）

```
src/
├── config/                  # PATH・定数・共通リスト
│   ├── paths.py             #   data/配下の各ドメインパス、PROJECT_ROOT等
│   ├── constants.py         #   PLACE_LIST, BETTING_TYPE_LIST等
│   └── lists.py
├── utils/
│   ├── file_utils.py         # CSV読み込み等の共通ヘルパー
│   └── format_data.py        # HTML生成で使う表データ整形（format_date, merge_rank_score等）
├── datasets/                 # Dataset層（構造・変換・バリデーション、I/Oなし）
│   ├── race_schedule/
│   ├── race_result/
│   ├── horse/
│   └── race_info/
├── managers/                  # Manager層（CSV読み書き・永続化）
│   ├── race_schedule_dataset_manager.py
│   ├── race_result_dataset_manager.py
│   ├── horse_peds_dataset_manager.py
│   ├── peds_results_dataset_manager.py
│   ├── past_performance_dataset_manager.py
│   ├── race_info_dataset_manager.py   # race_returnsも含む
│   ├── race_card_dataset_manager.py   # 出馬表+score/rank・per-raceレース情報・race_time_id_list
│   └── html_manager.py                # public_html/への書き出し・存在確認
├── logic/
│   ├── scraping/
│   │   ├── common.py             # HTTP/HTML共通処理
│   │   ├── netkeiba_scraper.py   # race_results / race_returns / horse_peds
│   │   └── jra_calendar_scraper.py # race_calendar
│   ├── scheduler/
│   │   ├── race_result_scheduler.py
│   │   ├── race_returns_scheduler.py
│   │   ├── horse_scheduler.py
│   │   └── race_info_scheduler.py
│   ├── prediction/
│   │   └── race_prediction_engine.py  # Oracle: 日次予想（LightGBM推論）
│   └── html_generator/                # Forge: HTML生成
│       ├── race_page_generator.py       # レース個別ページ
│       ├── horse_report_generator.py    # 出走馬詳細レポート
│       └── daily_index_generator.py     # 日次レース一覧ページ
└── output/
    └── prediction_publisher.py    # Herald: 予想テキスト生成・メール/X配信
```

データは `data/{race_schedule,race_result,horse,race_info,race_card}/` 配下に
開催場別・年別のCSVとして保存される（`src/config/paths.py` が単一の参照元）。

- `data/race_info/average_times/{place}/total_avg_time.csv` — コース・クラス別の
  平均走破タイム。Oracleの「過去走とのタイム差」特徴量で参照する
  （`race_info_dataset_manager.update_total_average_time(place_id, year)`で生成）。
- `data/race_info/{average_pops,average_weights,average_frames}/{place}/` と
  `data/race_info/average_times/{place}/total_winner_time.csv` — Forgeのレースページが
  表示する「コース別平均人気/馬体重/枠番・馬番/上り・通過」情報。2019〜2026年分を
  バックフィル済み（`race_info_scheduler.weekly_update_{average_pops,winners_weight,
  average_frame_and_horse,winner_time}`で生成・更新）。
- `data/race_card/{YYYYMMDD}/{race_id}.csv` — 出馬表+score/rank（旧 `RACE_CARDS_PATH`の
  新しい置き場所、`race_card_dataset_manager.save_race_cards`/`get_race_cards`）。
- `texts/race_prediction/{YYYYMMDD}/{race_id}.txt` — 予想テキスト（メール本文・Xポスト用、
  旧 `TEXT_PATH + "race_prediction/"` の新しい置き場所、`prediction_publisher.make_race_text`
  が生成）。

### public_html/（Forgeの出力先・Git管理対象）

```
public_html/
├── assets/
│   ├── css/styles.css
│   └── js/
└── races/
    └── {YYYYMMDD}/
        ├── index.html        # 日次レース一覧（daily_index_generator）
        └── {place}R{n}.html  # レース個別ページ（race_page_generator）
```

- `data/prediction/{models,datasets}/{place}/` — Oracleの学習済みLightGBMモデル・
  学習用データセット（旧 `data/PredictionModels/LightGBM/{Models,Datasets}/` を移動）。

## 3. 実際の動かし方

### 3-1. 前提

- プロジェクトルート（`keiba_ai_ver3.0/`）から実行する（`conftest.py` が `sys.path` に
  ルート・`libs/`・`src/legacy_datasets/` を追加する設定だが、新実装側
  `src/logic/scheduler/*` は `src.*` の絶対importのみで完結している）。
- race_result / race_returns / horse_peds の更新には netkeiba.com への通信が発生する。
- race_calendar の更新には JRA公式サイトへの通信が発生する。

### 3-2. 週次更新（直近1週間分の取り込み）

現状、本番エントリポイント（バッチ起動スクリプト等）への組み込みは未対応のため、
Pythonから各スケジューラ関数を直接呼び出す。

```python
from datetime import date
from src.logic.scheduler import (
    race_result_scheduler,
    race_returns_scheduler,
    horse_scheduler,
    race_info_scheduler,
)

day = date.today()

# race_result: 直近1週間のレース結果を取得し、保存・race_id別分割まで実行
race_result_scheduler.weekly_update_race_results(day)

# race_returns: 直近1週間の配当結果を取得し、保存・race_id別分割まで実行
race_returns_scheduler.weekly_update_race_returns(day)

# horse: 血統データ・血統別成績・出走馬の過去成績を更新
horse_scheduler.weekly_update_horse_peds(day)
horse_scheduler.weekly_update_pedsdata(day)
horse_scheduler.weekly_update_past_performance(day)

# race_info: race_resultを再集計（人気・馬体重・タイム等の平均値）
race_info_scheduler.weekly_update_horse_name_id_map(day.year)
race_info_scheduler.weekly_update_average_pops(day.year)
race_info_scheduler.weekly_update_winners_weight(day.year)
race_info_scheduler.weekly_update_average_frame_and_horse(day.year)
race_info_scheduler.weekly_update_winner_time(day.year)
race_info_scheduler.weekly_update_average_time(day.year)
```

`race_info_scheduler` の関数群は `data/race_result` の再集計のみで、ネットワーク通信は
発生しない（詳細は `specifications/WeeklyUpdateSequence.pu`）。

### 3-3. 月次更新 / 年初からの追い込み

```python
race_result_scheduler.monthly_update_race_results(day)
race_returns_scheduler.monthly_update_race_returns(day)
horse_scheduler.monthly_update_horse_peds(day)
horse_scheduler.monthly_update_pedsdata(day)
horse_scheduler.monthly_update_past_performance(day)
```

### 3-4. 過去データの一括作成（初期構築・年単位）

```python
# 2019年〜指定年までのデータを開催場別に取得して保存
race_result_scheduler.make_all_race_results(2026)
race_returns_scheduler.make_all_race_returns(2026)
horse_scheduler.make_all_horse_peds(2026)
horse_scheduler.make_all_pedsdata(2026)
horse_scheduler.make_all_past_performance(2026)
```

（一括/月次フローの詳細は `specifications/BatchCreationSequence.pu` を参照）

### 3-5. race_calendar（開催日程）の更新

`data/race_schedule/{year}_race_calendar.csv` が race_id算出の基礎データ。
JRA公式サイトから取得して保存する場合:

```python
from src.logic.scraping import jra_calendar_scraper

calendar_df = jra_calendar_scraper.get_jra_calendar(2026)
jra_calendar_scraper.save_race_calendar(2026, calendar_df)
```

現状どのスケジューラからも自動で呼ばれないため、年初等に手動で実行する想定。

### 3-6. 予想生成（Oracle）・配信

日次予想（出走馬のAI予想ランキング算出）は `src/logic/prediction/race_prediction_engine.py`
に移植済みで、`src/RacePrediction/day_race_prediction.py` の `rank_prediction(...)` から
呼び出される（シグネチャ・戻り値は旧実装と同一）。`race_card.py` 等の既存呼び出し元は
変更不要。

```python
from src.RacePrediction import day_race_prediction

rank_df = day_race_prediction.rank_prediction(race_id, horse_ids, race_info_df, waku_df)
```

オフライン学習パイプライン（`src/PredictionModels/LightGBM/`の学習・データセット作成系）は
今回のリファクタリング対象外。予想テキスト生成・メール/X配信は3-8を参照
（配当結果レポート・日次配信オーケストレーションは旧実装のまま、`bat/TodayRace/*.bat`等から
実行する）。

### 3-7. HTML生成（Forge）

レース個別ページ・日次レース一覧ページを `public_html/races/{YYYYMMDD}/` に生成する。

```python
from datetime import date
from src.logic.html_generator import race_page_generator, daily_index_generator

# 指定日の全レースページ（{place}R{n}.html）を生成
race_page_generator.make_daily_race_card_html(date.today())

# 指定日のレース一覧ページ（index.html）を生成
daily_index_generator.make_daily_index_page(date.today())
```

レースページは `data/race_card/{YYYYMMDD}/{race_id}.csv`（出馬表+score/rank）と
`data/race_info/{place}/{year}/{race_id}.csv`（レース情報）が存在するレースのみ
生成される。これらは `src/RacePrediction/race_card.py` / `make_time_id_list.py`
（出力先のみ新パスにリダイレクト済み）が日次予想時に生成する。

### 3-8. 予想テキスト生成・配信（Herald）

`src/output/prediction_publisher.py` が予想テキストの生成とメール/X(Twitter)配信を担う。
出馬表データセット（`data/race_card/{YYYYMMDD}/{race_id}.csv`、"rank"列が必要）から
予想テキストを生成し、`texts/race_prediction/{YYYYMMDD}/{race_id}.txt` に保存する。

```python
from datetime import date
from src.output import prediction_publisher

race_day = date.today()
race_id = "202404040601"

# 予想テキストを生成（texts/race_prediction/{YYYYMMDD}/{race_id}.txt に保存）
prediction_publisher.make_race_text(race_day, race_id)

# メール配信（.envのGMAIL_*設定でSMTP経由送信。実際にメールが送信される）
prediction_publisher.send_race_pred(race_day, race_id)

# X(Twitter)投稿（.envのX_*設定でtweepy経由投稿。実際に投稿される）
text_path = f"texts/race_prediction/{race_day.strftime('%Y%m%d')}/{race_id}.txt"
prediction_publisher.post_text_data(text_path)
```

`send_race_pred`/`post_text_data` はそれぞれ実際にメール送信・X投稿のAPI呼び出しを行うため、
動作確認時は注意する（`tests/test_prediction_publisher.py` では `smtplib.SMTP_SSL`/
`tweepy.Client` をmonkeypatchして単体テストしている）。

`libs/mail_api.py`・`libs/post_text.py`・`src/RacePrediction/make_text.py`の
`extract_top5_pred`/`make_race_text`は、後方互換のため `src.output.prediction_publisher`
への re-export として残置している。

## 4. テストの実行方法

`pytest.ini` で `testpaths = tests`、`network` マーカー（実サイト通信を伴うテスト）を
定義済み。`conftest.py` が `sys.path` を設定するため、プロジェクトルートで実行する。

```bash
# オフラインのみ（ネットワーク不要・CI向け）
pytest -m "not network"

# ネットワークテストのみ（netkeiba.com / JRA公式サイトへの実通信が発生）
pytest -m network

# 全テスト
pytest
```

### 新実装関連の主なテストファイル

| テストファイル | 内容 |
|---|---|
| `tests/test_race_schedule_dataset_manager.py` | race_schedule（Chronicle）の race_id 算出系、新旧出力比較 |
| `tests/test_jra_calendar_scraper.py`（network） | JRA開催カレンダー取得 |
| `tests/test_netkeiba_scraper.py`（一部 network） | race_results / race_returns スクレイピング、新旧比較・既知の確定結果との比較 |
| `tests/test_race_result_dataset_manager.py` | race_result の保存・分割・集計、新旧出力比較 |
| `tests/test_horse_peds_dataset_manager.py` | 血統データの取得・保存 |
| `tests/test_peds_results_dataset_manager.py` | 血統別成績の集計・保存 |
| `tests/test_past_performance_dataset_manager.py` | 出走馬の過去成績の再構築 |
| `tests/test_race_info_dataset_manager.py` | race_info系（人気・馬体重・タイム等）の集計、race_returnsの保存・分割 |
| `tests/test_race_returns_scheduler.py` | race_returns の週次/月次/一括更新オーケストレーション |
| `tests/test_race_prediction_engine.py` | Oracle（日次予想エンジン）の特徴量生成・LightGBM推論・新旧出力比較 |
| `tests/test_race_card_dataset_manager.py` | race_card（出馬表+score/rank・per-raceレース情報・race_time_id_list）の保存・取得 |
| `tests/test_horse_report_generator.py` | Forge: 出走馬詳細レポート（血統・近走・芝ダートサマリ）のHTML生成 |
| `tests/test_race_page_generator.py` | Forge: レース個別ページのHTML生成（コース別データ・出走馬レポート埋め込み） |
| `tests/test_daily_index_generator.py` | Forge: 日次レース一覧ページのHTML生成 |
| `tests/test_prediction_publisher.py` | Herald: 予想テキスト生成（`extract_top5_pred`/`make_race_text`）・メール/X配信のmonkeypatchテスト |

`tests/LightGBM_test.py`, `tests/dataset_test.py`, `tests/make_text_test.py`,
`tests/race_prediction_test.py` は旧実装（Forge/Herald相当・未移行部分）向けの既存テストで、
現在は実行可能なテスト関数を含まない（収集対象0件）。

## 5. 関連資料

- `specifications/新設計.md` — 目標アーキテクチャ
- `specifications/ContextDiagram.pu` 他 `specifications/*/ContextDiagram.pu` — 7モジュールのコンテキスト図
- `specifications/WeeklyUpdateSequence.pu` — 週次更新フロー（race_result / horse / race_info / race_returns）
- `specifications/BatchCreationSequence.pu` — 一括作成・月次更新フロー、race_calendar取得フロー
