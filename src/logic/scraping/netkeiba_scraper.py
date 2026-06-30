"""netkeiba.comからレース結果(race_result)を取得するスクレイパー

旧 libs/scraping.py の scrape_race_results / 旧 src/legacy_datasets/race_results.py の
scrape_race_results_dataframe を移植したもの。
HTTP/HTML共通処理は src/logic/scraping/common.py に切り出している。
"""

import re

import pandas as pd
from tqdm import tqdm

from src.config.constants import BETTING_TYPE_LIST
from src.datasets.race_card import transform as race_card_transform
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


def scrape_day_race_result(race_id):
    """race_idから、当日のレース結果を返す（race.netkeiba.comの速報ページを使用）

    旧 libs/scraping.py の scrape_day_race_results を移植したもの。
    db.netkeiba.com（scrape_race_results）と異なり、レース当日に結果が反映される
    速報ページを使うため、日次配信での即時結果取得に使う。

    Args:
        race_id (str): スクレイピングするrace_id

    Returns:
        pd.DataFrame: race_idのレース結果（インデックス = race_id、取得失敗時は空のDataFrame）
    """
    url = "https://race.netkeiba.com/race/result.html?race_id=" + str(race_id)
    if not common.url_exists(url):
        print("scrape_day_race_result: URL not found, skip", url)
        return pd.DataFrame()
    try:
        soup = common.fetch_soup(url)
        if not common.validate_soup(soup, url, "scrape_day_race_result", require_table=True):
            return pd.DataFrame()

        df_results = [pd.read_html(str(t))[0] for t in soup.select("table:has(tr td)")][0]
        df_results = df_results.rename(columns=lambda x: x.replace(" ", ""))
        df_results = transform.clean_day_race_results(df_results)
        df_results.index = [race_id] * len(df_results)
        return df_results
    except Exception as e:
        common.scraping_error(e)
        return pd.DataFrame()


# race.netkeiba.comの配当テーブル(Payout_Detail_Table)の行クラス名 → 式別名
_PAYOUT_ROW_BET_TYPE = {
    "Tansho": "単勝",
    "Fukusho": "複勝",
    "Wakuren": "枠連",
    "Umaren": "馬連",
    "Wide": "ワイド",
    "Umatan": "馬単",
    "Fuku3": "三連複",
    "Tan3": "三連単",
}
# 式別ごとの組み合わせの頭数（複勝/ワイドは結果が複数件あるため、この頭数ごとに区切る）
_PAYOUT_COMBO_SIZE = {
    "単勝": 1, "複勝": 1, "枠連": 2, "馬連": 2, "ワイド": 2, "馬単": 2, "三連複": 3, "三連単": 3,
}
# 着順を保つ必要がある式別（馬単・三連単）は"→"、それ以外は"-"で馬番を連結する
_PAYOUT_ORDERED_TYPES = {"馬単", "三連単"}


def scrape_day_race_returns(race_id):
    """race_idから、当日の配当結果を返す（race.netkeiba.comの速報ページを使用）

    db.netkeiba.com（scrape_race_returns_dataframe）は当該シーズン中のレースが
    まだ反映されておらず、当日〜近日中の配当結果を取得できない
    （ダミーの空ページが返ってくる）ため、レース結果と同じ速報ページ
    （race.netkeiba.com/race/result.html）内のPayout_Detail_Tableから取得する。

    Args:
        race_id (str): スクレイピングするrace_id

    Returns:
        pd.DataFrame: race_idの配当結果（列はrace_info_model.RACE_RETURNS_COLUMNS、
            インデックス = race_id。取得失敗時は空のDataFrame）
    """
    url = "https://race.netkeiba.com/race/result.html?race_id=" + str(race_id)
    if not common.url_exists(url):
        print("scrape_day_race_returns: URL not found, skip", url)
        return pd.DataFrame()
    try:
        soup = common.fetch_soup(url)
        if not common.validate_soup(soup, url, "scrape_day_race_returns", require_table=True):
            return pd.DataFrame()

        rows = []
        for table in soup.select("table.Payout_Detail_Table"):
            for tr in table.select("tr"):
                row_classes = tr.get("class") or []
                bet_type = next((_PAYOUT_ROW_BET_TYPE[c] for c in row_classes if c in _PAYOUT_ROW_BET_TYPE), None)
                if bet_type is None:
                    continue
                result_td = tr.select_one("td.Result")
                payout_td = tr.select_one("td.Payout")
                ninki_td = tr.select_one("td.Ninki")
                if result_td is None or payout_td is None or ninki_td is None:
                    continue

                numbers = [s.get_text(strip=True) for s in result_td.find_all("span") if s.get_text(strip=True)]
                payouts = [
                    p.strip().replace("円", "").replace(",", "")
                    for p in payout_td.get_text("|").split("|")
                    if p.strip()
                ]
                # 三連単等は組み合わせ数が多く、人気が「1,579人気」のようにカンマ区切りに
                # なることがあるため、配当と同様にカンマを取り除く
                ninkis = [
                    n.get_text(strip=True).replace("人気", "").replace(",", "")
                    for n in ninki_td.find_all("span")
                ]

                combo_size = _PAYOUT_COMBO_SIZE[bet_type]
                sep = "→" if bet_type in _PAYOUT_ORDERED_TYPES else "-"
                combos = [numbers[i:i + combo_size] for i in range(0, len(numbers), combo_size)]
                for combo, payout, ninki in zip(combos, payouts, ninkis):
                    rows.append({"式別": bet_type, "馬番": sep.join(combo), "配当": payout, "人気": ninki})

        df_returns = pd.DataFrame(rows, columns=race_info_model.RACE_RETURNS_COLUMNS)
        df_returns.index = [race_id] * len(df_returns)
        return df_returns
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


# netkeibaのh1.RaceName内、重賞アイコンのCSSクラス（Icon_GradeType{N}）のNから
# G1/G2/G3を判定するマッピング。実際の出馬表ページで安田記念・大阪杯・NHKマイルC
# （いずれもN=1）、京都記念・中山記念・チューリップ賞（いずれもN=2）、函館記念
# （N=3）で確認済み。N=13は「国際」マークで重賞の有無を問わず付くため対象外。
# Listed（N=15）・OP特別（N=17）等、G1/G2/G3以外のNは対象外（gradeはNoneのまま）。
_GRADE_TYPE_MAP = {"1": "G1", "2": "G2", "3": "G3"}


def _extract_race_grade(soup):
    """h1.RaceName内の重賞アイコン（Icon_GradeType{N}）からG1/G2/G3を判定する

    G1/G2/G3以外（Listed・OP特別等）や重賞アイコンが無いレースはNoneを返す。
    """
    h1 = soup.find("h1", attrs={"class": "RaceName"})
    if h1 is None:
        return None
    for span in h1.find_all("span", attrs={"class": "Icon_GradeType"}):
        for class_name in span.get("class", []):
            match = re.fullmatch(r"Icon_GradeType(\d+)", class_name)
            if match and match.group(1) in _GRADE_TYPE_MAP:
                return _GRADE_TYPE_MAP[match.group(1)]
    return None


def scrape_race_card(race_id):
    """race_idから、出馬表情報をスクレイピングする

    旧 libs/scraping.py の scrape_race_card を移植したもの。
    レース情報のトークン列からのrace_type/course_len/class/ground_state/weatherの
    抽出は src.datasets.race_card.transform.parse_race_card_info_tokens に切り出している。

    Args:
        race_id (str): race_id

    Returns:
        tuple: (info, race_info_df, race_card_df)
            info (list[str]): レース情報のトークン列
            race_info_df (pd.DataFrame): レース情報（race_type, course_len, class,
                ground_state, weather。1行。取得失敗時は空のDataFrame）
            race_card_df (pd.DataFrame): 出馬表（取得失敗時は空のDataFrame）
    """
    info = []
    try:
        url = "https://race.netkeiba.com/race/shutuba.html?race_id=" + str(race_id)
        if not common.url_exists(url):
            print("scrape_race_card: URL not found, skip", url)
            return info, pd.DataFrame(), pd.DataFrame()

        soup = common.fetch_soup(url)
        if not common.validate_soup(
            soup, url, "scrape_race_card", require_table=True,
            selectors=["h1.RaceName", "div.RaceData01", "div.RaceData02"],
        ):
            return info, pd.DataFrame(), pd.DataFrame()

        # メインとなるテーブルデータを取得
        df = pd.read_html(str(soup))[0]
        # 列名に半角スペースがあれば除去する
        df = df.rename(columns=lambda x: x.replace(" ", ""))
        # 後半部分を削除（オッズ・人気列までを残す。それより後ろは本登録/グループ等で不要）
        df = df.iloc[:, :11]
        df = df.drop(columns="印")
        # multicolumnを解除
        df.columns = df.columns.droplevel(0)
        # オッズ列はnetkeiba側のヘッダーセルが空でpandasに認識されないため、列位置で明示的に付け直す
        df.columns = list(df.columns[:-2]) + ["オッズ", "人気"]

        # レース情報を取得
        texts = (
            soup.find("h1", attrs={"class": "RaceName"}).text  # レース名
            + " "
            + soup.find("div", attrs={"class": "RaceData01"}).text  # 発走時刻
            + " "
            + soup.find("div", attrs={"class": "RaceData02"}).find_all("span")[3].text  # 馬齢
            + " "
            + soup.find("div", attrs={"class": "RaceData02"}).find_all("span")[4].text  # クラス
            + " "
            + soup.find("div", attrs={"class": "RaceData02"}).find_all("span")[5].text  # 種別
            + " "
            + soup.find("div", attrs={"class": "RaceData02"}).find_all("span")[6].text  # 斤量
            + " "
            + soup.find("div", attrs={"class": "RaceData02"}).find_all("span")[7].text  # 頭数
        )
        info = re.findall(r"\w+", texts)
        # Aコースなどの表記を消去
        info = [s for s in info if s not in ("A", "B", "C")]
        horse_numbers = int(re.findall(r"\d+", info[-1])[0])

        race_info_df = pd.DataFrame([race_card_transform.parse_race_card_info_tokens(info)])
        race_info_df["grade"] = _extract_race_grade(soup)

        # 厩舎名と所属を分離
        local = []
        for i in range(len(df)):
            if "美浦" in str(df.at[i, "厩舎"]):
                local.append("美浦")
                df.at[i, "厩舎"] = str(df.at[i, "厩舎"]).replace("美浦", "")
            elif "栗東" in str(df.at[i, "厩舎"]):
                local.append("栗東")
                df.at[i, "厩舎"] = str(df.at[i, "厩舎"]).replace("栗東", "")
            else:
                local.append(" ")
        df["所属"] = local

        # 馬ID、騎手IDをスクレイピング
        horse_id_list = []
        jockey_id_list = []
        horse_list = soup.find_all("tr", attrs={"class": "HorseList"})
        horse_count = 0
        for horse_infos in horse_list:
            horse_info = horse_infos.find("span", attrs={"class": "HorseName"}).find(
                "a", attrs={"href": re.compile("https")}
            )
            horse_id_list.append(re.findall(r"\d+", horse_info["href"])[0])

            jockey_info = horse_infos.find("td", attrs={"class": "Jockey"}).find(
                "a", attrs={"href": re.compile("https")}
            )
            jockey_id_list.append(re.findall(r"\d+", jockey_info["href"])[0])
            horse_count += 1
            if horse_count >= horse_numbers:
                break

        df["horse_id"] = horse_id_list
        df["jockey_id"] = jockey_id_list

        # インデックスをrace_idにする
        df.index = [race_id] * len(df)
        return info, race_info_df, df
    except Exception as e:
        common.scraping_error(e)
        return info, pd.DataFrame(), pd.DataFrame()


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
