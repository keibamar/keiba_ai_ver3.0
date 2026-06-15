# keiba_ai_ver3.0

netkeiba.com / JRA公式サイトからレースデータ・血統・配当などをスクレイピングして
データセットを蓄積し、出走馬の予想を行う競馬予想AIプロジェクト。

`specifications/新設計.md` で定義した新アーキテクチャ（`src/{datasets,managers,logic,
output,config,utils}` の6層構造 × 7モジュール）への移行を、ドメインごとに
段階的に進めている。

## 1. リファクタリングの進捗状況（2026-06-15時点）

**全体は未完了。** データ収集・蓄積系（Chronicle / Atlas / Reaper 相当）、
予想エンジン（Oracle）、HTML生成系（Forge）、配信系（Herald・予想テキスト生成＋配信、
配当結果レポート）、日次配信オーケストレーション（`post_daily_race.py`等）は
新構造への移行が完了しているが、旧ファイルのクリーンアップ（大部分）は未着手。

### 完了済み

| ドメイン | 旧実装 | 新実装 |
|---|---|---|
| race_schedule（Chronicle） | `libs/get_race_id.py` | `src/datasets/race_schedule/`, `src/managers/race_schedule_dataset_manager.py`, `src/logic/scraping/jra_calendar_scraper.py` |
| race_result（Atlas/Reaper） | `src/legacy_datasets/race_results.py`（`tests/test_race_result_dataset_manager.py`からの参照は解消済みだが、`average_time.py`/`past_performance.py`/`horse_peds.py`/`peds_results.py`がsibling importで参照するため残置） | `src/datasets/race_result/`, `src/managers/race_result_dataset_manager.py`, `src/logic/scraping/netkeiba_scraper.py`, `src/logic/scheduler/race_result_scheduler.py` |
| horse（Atlas） | `horse_peds.py` / `peds_results.py` / `past_performance.py` | `src/datasets/horse/`, `src/managers/{horse_peds,peds_results,past_performance}_dataset_manager.py`, `src/logic/scheduler/horse_scheduler.py` |
| race_info / race_returns（Atlas） | `analysis_race_info.py` / `analysis_race_time.py` / `average_time.py` / `race_returns.py` | `src/datasets/race_info/`, `src/managers/race_info_dataset_manager.py`, `src/logic/scheduler/{race_info_scheduler,race_returns_scheduler}.py` |
| config / utils | `libs/name_header.py` 等 | `src/config/{paths,constants,lists}.py`, `src/utils/file_utils.py` |
| 認証情報 | コード内ハードコード | `.env`（`libs/mail_api.py`, `libs/post_text.py` が読み込み） |
| 予想エンジン（Oracle・日次予想パスのみ） | `src/PredictionModels/LightGBM/{make_dataset,prediction}.py`, `src/RacePrediction/day_race_prediction.py`（削除済み） | `src/logic/prediction/race_prediction_engine.py` |
| race_card / HTML生成（Forge） | `src/RacePrediction/race_card.py` / `make_time_id_list.py`（出力先のみ）, `web/src/generators/{race_pages,horse_info,daily_index,make_race_card_html}.py` | `src/managers/{race_card_dataset_manager,html_manager}.py`, `src/logic/html_generator/{race_page_generator,horse_report_generator,daily_index_generator}.py`, `src/utils/format_data.py`, `public_html/` |
| race_card（出馬表生成） | `src/RacePrediction/race_card.py`（`make_race_card`/`extract_peds_for_display`）, ver2.0 `libs/scraping.py`（`scrape_race_card`） | `src/logic/prediction/race_card_builder.py`（`make_race_card`）, `src/datasets/race_card/transform.py`, `src/logic/scraping/netkeiba_scraper.py`（`scrape_race_card`） |
| 配当結果CSV保存（race_day_scheduler用） | `src/RacePrediction/calc_returns.py`（`get_race_return`/`save_each_race_return_csv`） | `src/logic/scraping/netkeiba_scraper.py`（`scrape_race_returns_dataframe`）, `src/managers/race_info_dataset_manager.py`（`save_race_return_for_race_id`） |
| カレンダー更新（race_day_scheduler用） | `web/src/generators/date_index.py`（`add_race_day`） | `src/managers/html_manager.py`（`add_race_day`、`public_html/assets/js/raceDays.js`を更新） |
| 予想テキスト生成・配信（Herald） | `src/RacePrediction/make_text.py`（`extract_top5_pred`/`make_race_text`）, `libs/{mail_api,post_text}.py` | `src/output/prediction_publisher.py` |
| 配当結果レポート（Herald残部） | `src/RacePrediction/calc_returns.py`（`get_win_result`/`get_place_result`/`get_trio_box_result`/`post_race_rerurns`）, `src/RacePrediction/make_text.py`（`write_{win,place,trio}_hit_text`/`make_return_text`） | `src/output/return_report.py` |
| 日次レース結果保存 | `src/RacePrediction/daily_race_results.py`（削除済み、`save_each_race_result_csv`/`save_day_race_result_each`/`get_each_race_results`） | `src/logic/scraping/netkeiba_scraper.py`（`scrape_day_race_result`）, `src/managers/race_result_dataset_manager.py`（`save_race_result_for_race_id`）, `src/logic/scheduler/race_result_scheduler.py`（`update_daily_race_results`） |
| 日次配信オーケストレーション | `src/RacePrediction/post_daily_race.py`（削除済み、`post_race_pred`/`post_pred_return`/`post_daily_race_pred`） | `src/logic/scheduler/race_day_scheduler.py` |
| average_calculator（平均タイム計算） | `src/datasets/race_info/transform.py`（`calc_avg_time`/`get_avg_time_list_from_race_results_df`/`make_avg_time_dataset`/`make_average_time_datasets`/`extract_course_race_results`/`get_race_time_msec`/`calc_time_diff`） | `src/logic/calculators/average_calculator.py` |

### 未対応（今後のフェーズ）

- **Oracle オフライン学習パイプライン**: `src/PredictionModels/LightGBM/`の`make_dataset_for_train`,
  `lightGBM_rank_train`, `prediction_rank`, `weekly_update_dataset_for_train`,
  `make_annual_dataset` 等は対象外（旧実装のまま、ver2.0データパスを参照し続ける）
- **race_card.pyのバッチ/CLI専用ロジック**: `src/RacePrediction/race_card.py`の
  `daily_race_card`/`get_race_info`（`__main__`のバッチ処理）、および
  `src/RacePrediction/make_time_id_list.py`（`get_race_id.get_daily_id`に依存）は、
  `race_day_scheduler`が直接使わないため対象外
- **calc_returns.pyのその他の孤立した回収率レポートCSV機能**: `get_quinella_box_result`/
  `get_quinella_wheel_result`/`get_trio_wheel_result`/`calc_day_race_return`/
  `calc_day_race_return_all`/`save_day_race_return_csv`/`save_calc_day_return`/
  `run_all_year`/`weekly_update_race_returns`
  （`race_returns_scheduler.weekly_update_race_returns`と重複）は、`make_return_text`からも
  `race_day_scheduler`からも呼ばれておらず`calc_returns.py`の`__main__`専用のため対象外。
  `name_header`/`get_race_id`/`scraping`/ver2.0の`race_card`/`race_returns`への依存は
  そのまま残置
- **クリーンアップ**: `libs/`, `src/legacy_datasets/`, `src/RacePrediction/`, `web/`等の削除（呼び出し元を新実装に切り替えた後）。
  以下のフェーズに分割して段階的に進める。
  - フェーズ1（完了）: `src/RacePrediction/{daily_race_results,post_daily_race,day_race_prediction}.py`
    （いずれも新実装への薄いリダイレクトのみ）と、対応する identity-check テスト
    （`tests/test_daily_race_results.py`/`tests/test_post_daily_race.py`）、
    `tests/test_race_prediction_engine.py`の旧実装比較テストを削除
  - フェーズ2（完了）: race_result系 — `tests/test_race_result_dataset_manager.py`の
    新旧比較を新実装単体のアサーションに置き換え済み。
    `src/legacy_datasets/race_results.py`は、`average_time.py`（フェーズ4）/
    `past_performance.py`/`horse_peds.py`/`peds_results.py`（フェーズ3、完了済みだが
    ファイル自体は残置）/`monthly_update.py`/`weekly_update.py`（フェーズ5）が
    内部で`import race_results`（sibling import）として`race_results.get_race_results_csv`を
    参照しているため削除できず、これらすべて（フェーズ4・5の対象）が削除されるまで残置する
  - フェーズ3（完了）: horse系 — 3ファイル合計899行と大きいため、ファイル単位でサブフェーズに分割
    - フェーズ3a（完了）: `tests/test_horse_peds_dataset_manager.py`の新旧比較を
      新実装単体のアサーションに置き換え済み
    - フェーズ3b（完了）: `tests/test_past_performance_dataset_manager.py`の新旧比較を
      新実装単体のアサーションに置き換え済み
    - フェーズ3c（完了）: `tests/test_peds_results_dataset_manager.py`の新旧比較を
      新実装単体のアサーションに置き換え済み

    `src/legacy_datasets/{horse_peds,past_performance,peds_results}.py`は、
    `monthly_update.py`/`weekly_update.py`/`src/RacePrediction/race_card.py`
    （いずれもフェーズ5対象）が直接`import horse_peds`/`import past_performance`/
    `import peds_results`として参照しているため削除できず、フェーズ5完了まで残置する
  - フェーズ4（完了）: race_info/race_returns/average系 —
    `tests/test_race_info_dataset_manager.py`（652行・56テスト）が
    `analysis_race_info`/`analysis_race_time`/`average_time`/`race_returns`の
    4legacyモジュールを参照し大きいため、ファイル内セクション単位でサブフェーズに分割
    - フェーズ4a（完了）: `tests/test_race_info_dataset_manager.py`の
      make_empty_record・analyze_*系（analysis_race_info.py由来）・
      analyze_winners系（analysis_race_time.py由来）の新旧比較を
      新実装単体のアサーションに置き換え済み
    - フェーズ4b（完了）: 同ファイルのhorse_id_map/update_average_pops/
      update_winners_weight/update_average_frame_and_horse・
      update_winner_time/update_annual_average_time/update_total_average_timeの
      新旧比較を新実装単体のアサーションに置き換え済み
    - フェーズ4c（完了）: 同ファイルのrace_returns系純粋関数・書き込み・読み込み、
      `tests/test_average_calculator.py`の新旧比較を新実装単体のアサーションに
      置き換え済み。`tests/test_netkeiba_scraper.py`のrace_returns部分・
      `tests/test_race_returns_scheduler.py`はすでに新実装単体のアサーション/
      既知の期待値比較になっており、変更不要だった

    フェーズ4完了に伴い`src/legacy_datasets/{analysis_race_info,analysis_race_time,
    average_time,race_returns}.py`の削除を検討したが、いずれも残置が必要:
    - `analysis_race_info.py`/`analysis_race_time.py`: `src/legacy_datasets/weekly_update.py`
      （フェーズ7対象）が`import analysis_race_info`/`import analysis_race_time`として
      直接参照しているため削除不可
    - `average_time.py`: `tests/test_race_prediction_engine.py`の`get_time_diff`新旧比較は
      フェーズ5で解消済みだが、`src/legacy_datasets/{monthly_update,weekly_update}.py`
      （フェーズ7対象）と`src/PredictionModels/LightGBM/make_dataset.py`が
      `import average_time`として直接参照しているため削除不可
    - `race_returns.py`: `tests/test_netkeiba_scraper.py`の
      `test_old_scrape_race_returns_dataframe_is_broken`
      （旧実装が単体で例外になることを示すドキュメント的テストで新旧比較ではないが、
      旧実装への参照が残る）が残るため削除不可
  - フェーズ5（完了）: race_card/prediction系 — `tests/test_race_card_builder.py`、
    `tests/test_race_prediction_engine.py`の残り（get_time_diff新旧比較）、
    `tests/test_netkeiba_scraper.py`のrace_card部分の新旧比較を新実装単体のアサーションに
    置き換え済み

    削除候補（`src/RacePrediction/{race_card,calc_returns,make_time_id_list,make_text}.py`、
    `src/legacy_datasets/{make_calender,monthly_update,weekly_update}.py`）を調査した結果:
    - `src/RacePrediction/{race_card,calc_returns,make_time_id_list,make_text}.py`:
      フェーズ5のテスト置き換えにより`tests/`からの参照は無くなったが、
      `web/src/generators/make_race_card_html.py`（フェーズ6対象）が
      `import race_card`/`import calc_returns`/`import make_time_id_list`として
      直接参照し、`make_text.py`はそれらから`import make_text`として参照されているため、
      フェーズ6完了まで削除不可
    - `src/legacy_datasets/make_calender.py`: ver3.0内のどこからも`import`されていない
      （`src/logic/scraping/jra_calendar_scraper.py`のdocstringにも未使用と明記済み）
    - `src/legacy_datasets/monthly_update.py`/`weekly_update.py`: ver3.0内のどの`.py`からも
      `import`されていない（`bat/Datasets/update_monthly.bat`/`update_weekly.bat`は
      `keiba_ai_ver2.0\src\Datasets\`側の同名スクリプトを呼んでおり、ver3.0の
      `src/legacy_datasets/`配下のコピーは参照されていない）
    - 上記3ファイルはフェーズ5の対象外で、フェーズ7「最終クリーンアップ」で
      `src/legacy_datasets/`残り全体の削除と合わせて整理する。ただし`weekly_update.py`が
      `analysis_race_info`/`analysis_race_time`/`race_results`/`horse_peds`/`peds_results`/
      `past_performance`を`import`していること自体が、これらモジュールの削除を妨げる
      唯一の参照元であるため、フェーズ7で`weekly_update.py`/`monthly_update.py`を削除すれば
      それらも削除可能になる
  - フェーズ6（完了）: web/系 — `tests/test_html_manager.py`のold_date_index部分
    （`test_add_race_day_*_matches_old`、`test_add_race_day_raises_if_array_missing`の
    旧実装側assert）の新旧比較を新実装単体のアサーションに置き換え済み。
    `tests/test_horse_report_generator.py`は調査の結果、`old_`を含むのは
    `test_load_horse_peds_dict_keys_match_old_format`というテスト名のみ
    （新実装の辞書アダプタが旧フォーマットの慣習に従っているかの確認で、
    旧モジュールへの参照や新旧比較ではない）で、既に新実装単体のアサーションのみ
    のため変更不要だった

    `web/`ディレクトリ（`web/src`・`web/site`）の削除を検討した結果:
    - 上記のテスト置き換えにより、`web/`配下はver3.0の`src/`・`tests/`から
      一切参照されなくなった（`src/`内の"web/src/..."への参照はすべて
      移植元を示すdocstringコメントのみ）
    - `bat/MakeHTML/{make_html_prev_day,update_daily_html}.bat`、
      `bat/TodayRace/today_race_rerturns.bat`は`web/`を参照するが、すべて
      `C:\keiba_ai\keiba_ai_ver2.0\web\...`（ver2.0側）を指しており、ver3.0の
      `web/`は無関係
    - `web/src/generators/make_race_card_html.py`が`src.RacePrediction`の
      `race_card`/`calc_returns`/`make_time_id_list`を直接importしていた
      （フェーズ5で削除をブロックしていた唯一の参照元）が、`web/`全体が
      不要になったことでこの参照も解消された。これにより
      `src/RacePrediction/{race_card,calc_returns,make_time_id_list,make_text}.py`
      も参照元が無くなり削除候補となる
    - `web/site/`は旧静的サイトの生成済みHTML（`public_html/`がその新しい置き場所）
    - `web/`・`src/RacePrediction/{race_card,calc_returns,make_time_id_list,
      make_text}.py`の実際の削除は、フェーズ7（最終クリーンアップ）で
      `src/legacy_datasets/`の残りと合わせてまとめて行う
  - フェーズ7（未着手）: 最終クリーンアップ — `libs/`（ver3.0配下のコピー）、
    `src/legacy_datasets/`・`src/RacePrediction/`・`web/`の残り全体を削除し、
    `conftest.py`のLIBS_PATH/LEGACY_DATASETS_PATHのsys.path注入を削除

→ **データ収集・蓄積（週次/月次/年次更新）、予想エンジンの日次予想パス（Oracle）、
出馬表生成（`race_card_builder.make_race_card`）、レースページ・日次インデックスのHTML生成
（Forge）、予想テキスト生成・メール/X配信・配当結果レポート
（Herald・`src/output/{prediction_publisher,return_report}.py`）、
配当結果CSV保存（`netkeiba_scraper.scrape_race_returns_dataframe` /
`race_info_dataset_manager.save_race_return_for_race_id`）、カレンダー更新
（`html_manager.add_race_day`）、日次配信オーケストレーション本体
（`src/logic/scheduler/race_day_scheduler.py`）は新実装で動かせる**。
**実運用は現状`bat/TodayRace/post_today_race.bat`がver2.0側の`post_daily_race.py`を
呼んでおり、このリファクタリングによる影響はない。**

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
│   ├── race_info/
│   └── race_card/
├── managers/                  # Manager層（CSV読み書き・永続化）
│   ├── race_schedule_dataset_manager.py
│   ├── race_result_dataset_manager.py
│   ├── horse_peds_dataset_manager.py
│   ├── peds_results_dataset_manager.py
│   ├── past_performance_dataset_manager.py
│   ├── race_info_dataset_manager.py   # race_returnsも含む
│   ├── race_card_dataset_manager.py   # 出馬表+score/rank・per-raceレース情報・race_time_id_list
│   └── html_manager.py                # public_html/への書き出し・存在確認・カレンダー更新（add_race_day）
├── logic/
│   ├── scraping/
│   │   ├── common.py             # HTTP/HTML共通処理
│   │   ├── netkeiba_scraper.py   # race_results / race_returns / horse_peds
│   │   └── jra_calendar_scraper.py # race_calendar
│   ├── scheduler/
│   │   ├── race_result_scheduler.py
│   │   ├── race_returns_scheduler.py
│   │   ├── horse_scheduler.py
│   │   ├── race_info_scheduler.py
│   │   └── race_day_scheduler.py      # 日次配信オーケストレーション
│   ├── calculators/
│   │   └── average_calculator.py      # 平均タイム・タイム差の計算
│   ├── prediction/
│   │   ├── race_prediction_engine.py  # Oracle: 日次予想（LightGBM推論）
│   │   └── race_card_builder.py       # 出馬表生成（make_race_card）
│   └── html_generator/                # Forge: HTML生成
│       ├── race_page_generator.py       # レース個別ページ
│       ├── horse_report_generator.py    # 出走馬詳細レポート
│       └── daily_index_generator.py     # 日次レース一覧ページ
└── output/
    ├── prediction_publisher.py    # Herald: 予想テキスト生成・メール/X配信
    └── return_report.py           # Herald残部: 配当結果レポート（回収率テキスト生成・X配信）
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
- `data/race_result/{place}/{year}/{race_id}.csv` — 1レース分のレース結果（per-race CSV）。
  `race_result_dataset_manager.split_race_results_by_year`（年次データの分割）と
  `save_race_result_for_race_id`（速報ページからの即時保存、`race_result_scheduler.
  update_daily_race_results`が使用）の両方がこの構成に出力する。
- `data/race_card/{YYYYMMDD}/{race_id}.csv` — 出馬表+score/rank（旧 `RACE_CARDS_PATH`の
  新しい置き場所、`race_card_dataset_manager.save_race_cards`/`get_race_cards`）。
- `texts/race_prediction/{YYYYMMDD}/{race_id}.txt` — 予想テキスト（メール本文・Xポスト用、
  旧 `TEXT_PATH + "race_prediction/"` の新しい置き場所、`prediction_publisher.make_race_text`
  が生成）。
- `texts/race_returns/{YYYYMMDD}/{place}_pred_score.txt` — 配当結果レポート（回収率テキスト、
  旧 `TEXT_PATH + "race_returns/"` の新しい置き場所、`return_report.make_return_text`が生成）。

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
に移植済み。`src/RacePrediction/race_card.py` 等の既存呼び出し元はこの
`race_prediction_engine.rank_prediction(...)` を直接呼び出す。

```python
from src.logic.prediction import race_prediction_engine

rank_df = race_prediction_engine.rank_prediction(race_id, horse_ids, race_info_df, waku_df)
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

### 3-9. 配当結果レポート生成・配信（Herald残部）

`src/output/return_report.py` が、指数1位馬の的中率・回収率テキストの生成とX(Twitter)配信を担う。
出馬表データセット（`data/race_card/{YYYYMMDD}/{race_id}.csv`、"rank"列が必要）と
配当結果（`data/race_info/race_returns/{place}/{year}/{race_id}.csv`）から
レポートを生成し、`texts/race_returns/{YYYYMMDD}/{place}_pred_score.txt` に保存する。

```python
from datetime import date
from src.output import return_report

race_day = date.today()
place_id = 4  # 新潟

# 回収率レポートを生成（texts/race_returns/{YYYYMMDD}/{place}_pred_score.txt に保存）
return_report.make_return_text(place_id, race_day)

# X(Twitter)投稿（.envのX_*設定でtweepy経由投稿。実際に投稿される）
return_report.post_race_returns(place_id, race_day)
```

`src/RacePrediction/make_text.py`の`write_{win,place,trio}_hit_text`/`make_return_text`、
`src/RacePrediction/calc_returns.py`の`get_win_result`/`get_place_result`/
`get_trio_box_result`/`post_race_rerurns`（typo）は、後方互換のため
`src.output.return_report`への re-export として残置している。

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
| `tests/test_netkeiba_scraper.py`（一部 network） | race_results / 当日速報結果（scrape_day_race_result）スクレイピングの新旧比較、race_returns / 出馬表（scrape_race_card）スクレイピングの既知の確定結果との比較 |
| `tests/test_race_result_dataset_manager.py` | race_result の保存・分割・集計・per-race結果保存（save_race_result_for_race_id）の新実装単体検証 |
| `tests/test_horse_peds_dataset_manager.py` | 血統データの取得・保存・名前正規化の新実装単体検証 |
| `tests/test_peds_results_dataset_manager.py` | 血統別成績の集計・取得・保存・更新の新実装単体検証 |
| `tests/test_past_performance_dataset_manager.py` | 出走馬の過去成績の再構築・正規化・取得・保存の新実装単体検証 |
| `tests/test_race_info_dataset_manager.py` | race_info系（人気・馬体重・タイム等）の集計・race_returns系の純粋関数・保存・分割・取得（horse_id_map/update_\*系・race_returns系を含め新実装単体検証済み）、per-race配当結果保存（save_race_return_for_race_id） |
| `tests/test_race_returns_scheduler.py` | race_returns の週次/月次/一括更新オーケストレーション |
| `tests/test_race_result_scheduler.py` | race_result の日次結果取得オーケストレーション（update_daily_race_results） |
| `tests/test_race_day_scheduler.py` | 日次配信オーケストレーション（post_race_pred/post_pred_return のテキストパス組み立て・X投稿連携） |
| `tests/test_average_calculator.py` | average_calculator（平均タイム算出・タイム差計算）の新実装単体検証 |
| `tests/test_race_prediction_engine.py` | Oracle（日次予想エンジン）の特徴量生成・LightGBM推論・get_time_diffの新実装単体検証 |
| `tests/test_race_card_dataset_manager.py` | race_card（出馬表+score/rank・per-raceレース情報・race_time_id_list）の保存・取得 |
| `tests/test_race_card_builder.py`（一部 network） | 出馬表生成（`race_card_builder.make_race_card`、`race_card/transform.py`の各純粋関数）の新実装単体検証（make_race_cardは既知の期待値との比較） |
| `tests/test_html_manager.py` | カレンダー更新（`html_manager.add_race_day`）の新実装単体検証（新規作成・追記・重複時のno-op・不正フォーマット時のエラー） |
| `tests/test_horse_report_generator.py` | Forge: 出走馬詳細レポート（血統・近走・芝ダートサマリ）のHTML生成 |
| `tests/test_race_page_generator.py` | Forge: レース個別ページのHTML生成（コース別データ・出走馬レポート埋め込み） |
| `tests/test_daily_index_generator.py` | Forge: 日次レース一覧ページのHTML生成 |
| `tests/test_prediction_publisher.py` | Herald: 予想テキスト生成（`extract_top5_pred`/`make_race_text`）・メール/X配信のmonkeypatchテスト |
| `tests/test_return_report.py` | Herald残部: 配当結果レポート（`get_{win,place,trio_box}_result`/`make_return_text`/`post_race_returns`）のフィクスチャベーステスト |

## 5. 関連資料

- `specifications/新設計.md` — 目標アーキテクチャ
- `specifications/ContextDiagram.pu` 他 `specifications/*/ContextDiagram.pu` — 7モジュールのコンテキスト図
- `specifications/WeeklyUpdateSequence.pu` — 週次更新フロー（race_result / horse / race_info / race_returns）
- `specifications/BatchCreationSequence.pu` — 一括作成・月次更新フロー、race_calendar取得フロー
