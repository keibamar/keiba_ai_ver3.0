"""netkeiba.comからレース結果(race_result)を取得するスクレイパー

旧 libs/scraping.py の scrape_race_results / 旧 src/legacy_datasets/race_results.py の
scrape_race_results_dataframe を移植したもの。
HTTP/HTML共通処理は src/logic/scraping/common.py に切り出している。
"""

import re

import pandas as pd
from tqdm import tqdm

from src.config.constants import BETTING_TYPE_LIST
from src.datasets.race_info import model as race_info_model
from src.datasets.race_info import transform as race_info_transform
from src.datasets.race_result import model, transform
from src.logic.scraping import common


def scrape_race_results(race_id):
    """race_idから、レース結果、レース情報、horse_id/jockey_idを統合して返す

    Args:
        race_id (str): スクレイピングするrace_id

    Returns:
        pd.DataFrame: race_idのレース結果（インデックス = race_id）
    """
    try:
        url = "https://db.netkeiba.com/race/" + str(race_id)
        if not common.url_exists(url):
            print("scrape_race_results: URL not found, skip", url)
            return pd.DataFrame()

        soup = common.fetch_soup(url)
        if not common.validate_soup(
            soup, url, "scrape_race_results", require_table=True,
            selectors=["div.data_intro", 'table[summary="レース結果"]'],
        ):
            return pd.DataFrame()

        # テーブルの行を取得
        rows = soup.find_all("tr")
        expected_columns = len(model.RAW_COLUMNS)

        filtered_data = []
        for row in rows:
            cols = row.find_all("td")
            cols = [col.get_text(strip=True) for col in cols]
            if len(cols) == expected_columns:
                filtered_data.append(cols)

        df_results = pd.DataFrame(filtered_data, columns=model.RAW_COLUMNS)

        # 列名に半角スペースがあれば除去する
        df_results = df_results.rename(columns=lambda x: x.replace(" ", ""))

        for i in range(len(df_results)):
            # 時間表記を変更
            if df_results.notnull().at[i, "タイム"]:
                df_results.at[i, "タイム"] = "0:" + df_results.at[i, "タイム"]
            # 馬体重増減削除
            if df_results.notnull().at[i, "馬体重"]:
                temp = df_results.at[i, "馬体重"]
                pattern = r"\([^()]*\)"
                df_results.at[i, "馬体重"] = re.sub(pattern, "", temp)

        # 天候、レースの種類、コースの長さ、馬場の状態、日付を取得する
        texts = (
            soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
            + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
        )
        tokens = re.findall(r"\w+", texts)
        info = transform.parse_race_info_tokens(tokens)
        for key, value in info.items():
            df_results[key] = [value] * len(df_results)

        # course_lenを整数型に統一（floatになってしまうケースに対応）
        if "course_len" in df_results.columns:
            try:
                df_results["course_len"] = int(df_results["course_len"].astype(float))
            except Exception:
                df_results["course_len"] = df_results["course_len"].apply(
                    lambda x: int(x) if pd.notnull(x) else x
                )

        # 馬ID、騎手IDをスクレイピング
        horse_id_list = []
        horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
            "a", attrs={"href": re.compile("^/horse")}
        )
        for a in horse_a_list:
            horse_id = re.findall(r"\d+", a["href"])
            horse_id_list.append(horse_id[0])
        jockey_id_list = []
        jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
            "a", attrs={"href": re.compile("^/jockey")}
        )
        for a in jockey_a_list:
            jockey_id = re.findall(r"\d+", a["href"])
            jockey_id_list.append(jockey_id[0])
        df_results["horse_id"] = horse_id_list
        df_results["jockey_id"] = jockey_id_list

        # 不要列を削除
        df_results = df_results.drop(
            columns=[c for c in model.DROP_COLUMNS if c in df_results.columns], errors="ignore"
        )
        # インデックスをrace_idにする
        df_results.index = [race_id] * len(df_results)
        return df_results
    except Exception as e:
        common.scraping_error(e)
        return pd.DataFrame()


def scrape_horse_peds(horse_id):
    """horse_idから血統データを取得する

    旧 src/legacy_datasets/horse_peds.py の make_horse_peds_dataset を移植したもの。

    Args:
        horse_id (str): horse_id

    Returns:
        pd.Series: 列名 peds_0..peds_61、Series名がhorse_idの血統データ
            （取得に失敗した場合は空のDataFrame）
    """
    url = "https://db.netkeiba.com/horse/ped/" + str(horse_id)
    try:
        tables = common.scrape_df(url)
        if not tables:
            return pd.DataFrame()

        peds_df = tables[0]
        # 重複を削除して1列のSeries型データに直す
        generations = {}
        for i in reversed(range(5)):
            generations[i] = peds_df[i]
            peds_df = peds_df.drop([i], axis=1)
            peds_df = peds_df.drop_duplicates()
        ped = pd.concat([generations[i] for i in range(5)]).rename(str(horse_id))
        # インデックスをpeds_0, ..., peds_61にする
        return ped.reset_index(drop=True).T.add_prefix("peds_")
    except Exception as e:
        common.scraping_error(e)
        return pd.DataFrame()


def scrape_race_results_dataframe(race_id_list):
    """race_id_listのrace_resultsのDataFrameを作成

    Args:
        race_id_list (list[str]): race_idのリスト

    Returns:
        pd.DataFrame: race_id_listのrace_results
    """
    race_results_df = pd.DataFrame()
    for race_id in tqdm(race_id_list):
        new_df = scrape_race_results(race_id)
        new_df = transform.clean_columns(new_df)
        base_columns = list(dict.fromkeys(list(race_results_df.columns) + list(new_df.columns)))
        new_df = new_df.reindex(columns=base_columns)
        race_results_df = pd.concat([race_results_df, new_df], axis=0)
    return race_results_df


def scrape_race_returns_dataframe(race_id_list):
    """race_id_listの配当結果(race_returns)のDataFrameを作成

    旧 src/legacy_datasets/race_returns.py の scrape_race_returns_dataframe を移植したもの。

    旧実装は、format_type_returns_dataframeの戻り値（列名 "式別","馬番","配当","人気"）に対して
    .set_index(0) を呼び出しており、存在しない列"0"を指定するためKeyErrorとなり処理全体が
    失敗するバグがあった。新実装ではこの呼び出しを行わず、format_type_returns_dataframeの
    戻り値をそのまま連結する。

    Args:
        race_id_list (list[str]): race_idのリスト

    Returns:
        pd.DataFrame: race_id_listのrace_returns（インデックス = race_id、
            列は src.datasets.race_info.model.RACE_RETURNS_COLUMNS）
    """
    return_tables = {}
    for race_id in tqdm(race_id_list):
        url = "https://db.netkeiba.com/race/" + str(race_id)
        tables = common.scrape_df(url)
        race_return_df = race_info_transform.extract_race_return_table(tables)
        race_return_df = race_info_transform.format_race_return_dataframe(race_return_df)
        if race_return_df.empty:
            continue

        type_dfs = [
            race_info_transform.format_type_returns_dataframe(race_return_df, bet_type)
            for bet_type in BETTING_TYPE_LIST
            if race_info_transform.type_check(race_return_df, bet_type)
        ]
        if not type_dfs:
            continue

        race_returns_df = pd.concat(type_dfs, ignore_index=True)
        race_returns_df.index = [race_id] * len(race_returns_df)
        return_tables[race_id] = race_returns_df

    if not return_tables:
        return pd.DataFrame(columns=race_info_model.RACE_RETURNS_COLUMNS)

    return pd.concat([return_tables[key] for key in return_tables])
