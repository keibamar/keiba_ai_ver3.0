import re
import sys
from time import time
import warnings

from bs4 import BeautifulSoup
import pandas as pd
import requests

# pycache を生成しない
sys.dont_write_bytecode = True
# エラー表記をしない
warnings.simplefilter('ignore')

scraping_header =  {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.3'
}


def url_exists(url: str) -> bool:
    """URLが存在するかどうかを簡易チェックする。
    HEAD を試み、ダメなら GET にフォールバックしてステータスコードを確認する。
    タイムアウトや例外が発生した場合は False を返す。
    """
    try:
        # まずはHEADで存在チェック（軽量）
        resp = requests.head(url, headers=scraping_header, allow_redirects=True, timeout=5)
        if resp.status_code == 200:
            return True
        # 一部サイトはHEADを拒否するためGETでフォールバック
        resp = requests.get(url, headers=scraping_header, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def validate_soup(soup, url: str, func_name: str, require_table: bool = False, selectors: list | None = None) -> bool:
    """soupの中身が期待通りかをチェックする。問題があればログ出力してFalseを返す。

    Args:
        soup: BeautifulSoupオブジェクト
        url (str): チェック対象のURL（ログ用）
        func_name (str): 呼び出し関数名（ログ用）
        require_table (bool): テーブルが必須か
        selectors (list[str]|None): 期待するCSSセレクタのリスト

    Returns:
        bool: 有効ならTrue、無ければFalse
    """
    try:
        if not soup or not getattr(soup, 'text', '').strip():
            print(f"{func_name}: empty or no HTML content, skip {url}")
            return False
        if require_table and not soup.find('table'):
            print(f"{func_name}: no <table> found, skip {url}")
            return False
        if selectors:
            for sel in selectors:
                if not soup.select_one(sel):
                    print(f"{func_name}: expected selector '{sel}' not found, skip {url}")
                    return False
        return True
    except Exception as e:
        print(f"{func_name}: validate_soup error {e.__class__.__name__}: {e} for {url}")
        return False

def scraping_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def scrape_df(url):
    """ urlからスクレイピングをし、DataFrameを返す 
        Args:
            url (str) : スクレイピングするurl

        Returns:
            DataFrame: url先のテーブルデータをDataFrame型で返す    
        """
    # URL存在チェック
    if not url_exists(url):
        print("scrape_df: URL not found, skip", url)
        return pd.DataFrame()

    try:
        # urlからスクレイピング
        html = requests.get(url, headers=scraping_header)
        html.encoding = "EUC-JP"
        soup = BeautifulSoup(html.text, "html.parser")
        # バグ対策でdecode
        soup = BeautifulSoup(html.content.decode("euc-jp", "ignore"), "html.parser")
        # soupの基本チェック
        if not validate_soup(soup, url, 'scrape_df', require_table=True):
            return pd.DataFrame()
        
        # テーブルデータを取得
        df = [pd.read_html(str(t))[0] for t in soup.select('table:has(tr td)')]
        if not df:
            print("scrape_error: No table found ", url)
        return df
    
    except Exception as e:
        scraping_error(e)
        return pd.DataFrame()

def scrape_race_results(race_id):
    """ race_idから、レース結果、レース情報、horse_idを統合して返す 
        Args:
            race_id (str) : スクレイピングするrace_id

        Returns:
            DataFrame: url先のレース結果をレース情報と、horse_idを統合したDataFrame型で返す    
    """                     
    try:
        url = "https://db.netkeiba.com/race/" + str(race_id)
        # URL存在チェック
        if not url_exists(url):
            print("scrape_race_results: URL not found, skip", url)
            return pd.DataFrame()
        # urlからスクレイピング
        html = requests.get(url, headers=scraping_header)
        html.encoding = "EUC-JP"
        soup = BeautifulSoup(html.text, "html.parser")
        # バグ対策でdecode
        soup = BeautifulSoup(html.content.decode("euc-jp", "ignore"), "html.parser")
        # soupの基本チェック
        if not validate_soup(soup, url, 'scrape_race_results', require_table=True, selectors=['div.data_intro','table[summary="レース結果"]']):
            return pd.DataFrame()
        # テーブルの行を取得    
        rows = soup.find_all('tr')
        expected_columns = 25  # 期待する列数

        filtered_data = []
        for row in rows:
            cols = row.find_all('td')
            cols = [col.get_text(strip=True) for col in cols]
            if len(cols) == expected_columns:
                filtered_data.append(cols)

        # データフレームに変換
        df_results = pd.DataFrame(filtered_data, columns=[
            "着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手",
            "タイム", "着差", "タイム指数", "タイム指数2","タイム指数3","タイム指数4","タイム指数5","通過", "上り", "単勝",
            "人気", "馬体重", "調教タイム", "厩舎コメント", "備考", "調教師", "馬主", "賞金"
        ])

        # メインとなるテーブルデータを取得
        # df_results = [pd.read_html(str(t))[0] for t in soup.select('table:has(tr td)')][0]
        # 列名に半角スペースがあれば除去する
        df_results = df_results.rename(columns=lambda x: x.replace(' ', ''))
        # print(df_results)
        for i in range(len(df_results)):
            # 時間表記を変更
            if df_results.notnull().at[i,"タイム"]:
                df_results.at[i,"タイム"] = "0:" + df_results.at[i,"タイム"]
            # 馬体重増減削除
            if df_results.notnull().at[i,"馬体重"]:
                temp = df_results.at[i,"馬体重"]
                # 括弧とその中身を含む正規表現パターンを定義する
                pattern = r'\([^()]*\)'
                # 正規表現パターンに一致する部分を空文字列で置換する
                df_results.at[i,"馬体重"] = re.sub(pattern, '', temp)

        # 天候、レースの種類、コースの長さ、馬場の状態、日付を取得する
        texts = (
            soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
            + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
        )
        info = re.findall(r'\w+', texts)

        for text in info:
            # print(text)
            if text in ["芝", "ダート"]:
                df_results["race_type"] = [text] * len(df_results)
            if "障" in text:
                df_results["race_type"] = ["障害"] * len(df_results)
            if "m" in text:
                df_results["course_len"] = [int(re.findall(r"\d+", text)[-1])] * len(df_results)
            if text in ["2歳新馬","3歳新馬"]:
                df_results["class"] = ["新馬"] * len(df_results)
            if text in ["2歳未勝利","3歳未勝利","障害3歳以上未勝利","障害4歳以上未勝利"]:
                df_results["class"] = ["未勝利"] * len(df_results)
            if text in ["2歳1勝クラス","3歳1勝クラス","3歳以上1勝クラス","4歳以上1勝クラス"]:
                df_results["class"] = ["1勝クラス"] * len(df_results)
            if text in ["3歳2勝クラス","3歳以上2勝クラス","4歳以上2勝クラス"]:
                df_results["class"] = ["2勝クラス"] * len(df_results)   
            if text in ["3歳以上3勝クラス","4歳以上3勝クラス"]:
                df_results["class"] = ["3勝クラス"] * len(df_results)    
            if text in ["2歳オープン","3歳オープン","3歳以上オープン","4歳以上オープン","障害3歳以上オープン","障害4歳以上オープン"]:
                df_results["class"] = ["オープン"] * len(df_results)         
            if text in ["良", "稍重", "重", "不良"]:
                df_results["ground_state"] = [text] * len(df_results)
            if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                df_results["weather"] = [text] * len(df_results)
            if "年" in text:
                df_results["date"] = [text] * len(df_results)
        # course_lenを整数型に統一（floatになってしまうケースに対応）
        if "course_len" in df_results.columns:
            try:
                df_results["course_len"] = int(df_results["course_len"].astype(float))
            except Exception:
                df_results["course_len"] = df_results["course_len"].apply(lambda x: int(x) if pd.notnull(x) else x)
        
        #馬ID、騎手IDをスクレイピング
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

        # --- 不要列 ---
        drop_columns = ["タイム指数","タイム指数2","タイム指数3","タイム指数4","タイム指数5", "調教タイム", "厩舎コメント", "備考", "馬主", "賞金"]
        df_results = df_results.drop(columns=[c for c in drop_columns if c in df_results.columns], errors="ignore")
        #インデックスをrace_idにする
        df_results.index = [race_id] * len(df_results)
        print(df_results)
        return df_results
    except Exception as e:
        scraping_error(e)
        return pd.DataFrame()


def clean_race_results(df: pd.DataFrame):
    """
        レース結果の体裁を整える。
    """
    df = df.copy()

    for i in range(len(df)):
        # 時間表記を変更
        if df.notnull().at[i,"タイム"]:
            df.at[i,"タイム"] = "0:" + df.at[i,"タイム"]

    # --- 列名変更 ---
    df = df.rename(columns={
        "後3F": "上り",
        "コーナー通過順": "通過",
        "単勝オッズ": "単勝",
        "馬体重(増減)": "馬体重"
    })

    # --- 列順を見やすく調整（任意） ---
    reorder_cols = [
        "着順", "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差",
        "人気", "単勝", "上り", "通過", "厩舎", "馬体重"
    ]
    df = df[[c for c in reorder_cols if c in df.columns]]

    return df
    
def scrape_day_race_results(race_id):
    """ race_idから、当日のレース結果結果を返す 
        Args:
            race_id (str) : スクレイピングするrace_id

        Returns:
            df_return(pd.DataFrame) : race_idのレース結果結果    
    """
    # 配当結果の取得
    url = "https://race.netkeiba.com/race/result.html?race_id=" + race_id
    # URL存在チェック
    if not url_exists(url):
        print("scrape_day_race_results: URL not found, skip", url)
        return pd.DataFrame()
    html = requests.get(url, headers=scraping_header)
    html.encoding = "EUC-JP"
    try:
        soup = BeautifulSoup(html.text, "html.parser")
        # バグ対策でdecode
        soup = BeautifulSoup(html.content.decode("euc-jp", "ignore"), "html.parser")
        if not validate_soup(soup, url, 'scrape_day_race_results', require_table=True):
            return pd.DataFrame()
        
        # メインとなるテーブルデータを取得
        df_results = [pd.read_html(str(t))[0] for t in soup.select('table:has(tr td)')][0]
        # 列名に半角スペースがあれば除去する
        df_results = df_results.rename(columns=lambda x: x.replace(' ', ''))
        
        df_results = clean_race_results(df_results)

        #インデックスをrace_idにする
        df_results.index = [race_id] * len(df_results)

        return df_results
    except Exception as e:
        scraping_error(e)
        return pd.DataFrame()

def scrape_day_race_returns(race_id):
    """ race_idから、当日の配当結果を返す 
        Args:
            race_id (str) : スクレイピングするrace_id

        Returns:
            df_return(pd.DataFrame) : race_idの配当結果    
    """
    # 配当結果の取得
    url = "https://race.netkeiba.com/race/result.html?race_id=" + race_id
    # URL存在チェック
    if not url_exists(url):
        print("scrape_day_race_returns: URL not found, skip", url)
        return pd.DataFrame()
    html = requests.get(url, headers=scraping_header)
    html.encoding = "EUC-JP"

    try:
        # soupで中身チェック（pd.read_html前）
        soup = BeautifulSoup(html.text, "html.parser")
        soup = BeautifulSoup(html.content.decode("euc-jp", "ignore"), "html.parser")
        if not validate_soup(soup, url, 'scrape_day_race_returns', require_table=True):
            return pd.DataFrame()
        # 単勝、複勝、枠連、馬連のデータ
        df1 = pd.read_html(html.text)[1]
        # ワイド、馬単、三連複、三連単のデータ
        df2 = pd.read_html(html.text)[2]
        df_return = pd.concat([df1,df2]).reset_index(drop = True)
        return df_return
    
    except Exception as e:
        scraping_error(e)
        return pd.DataFrame()

def scrape_race_card(race_id):
    """ race_idから、出馬表情報をスクレイピング
        Args:
            race_id (str) : race_id

        Returns:
            info(list) : レース情報
            df_info(pd.DataFrame) : レース情報
            df(pd.DataFrame) : 出馬表
    """   
    try :
        info =[]
        url = "https://race.netkeiba.com/race/shutuba.html?race_id=" + str(race_id)
        # URL存在チェック
        if not url_exists(url):
            print("scrape_race_card: URL not found, skip", url)
            return info, pd.DataFrame(), pd.DataFrame()
        # スクレイピング
        html = requests.get(url, headers=scraping_header)
        html.encoding = "EUC-JP"

        # soupチェック
        soup = BeautifulSoup(html.text, "html.parser")
        soup = BeautifulSoup(html.content.decode("euc-jp", "ignore"), "html.parser")
        if not validate_soup(soup, url, 'scrape_race_card', require_table=True, selectors=['h1.RaceName','div.RaceData01','div.RaceData02']):
            return info, pd.DataFrame(), pd.DataFrame()

        # メインとなるテーブルデータを取得
        df = pd.read_html(html.text)[0]

        # 列名に半角スペースがあれば除去する
        df = df.rename(columns=lambda x: x.replace(' ', ''))

        # 後半部分を削除
        df = df.iloc[:,:9]
        df = df.drop(columns = '印')
        # multicolumを解除
        df.columns = df.columns.droplevel(0)
        # レース情報をスクレイピング
        soup = BeautifulSoup(html.text, "html.parser")
        texts = (
            soup.find("h1", attrs={"class": "RaceName"}).text                              # レース名
            + " "
            + soup.find("div", attrs={"class": "RaceData01"}).text                          # 発走時刻
            + " "
            + soup.find("div", attrs={"class": "RaceData02"}).find_all("span")[3].text      # 馬齢
            + " "
            + soup.find("div", attrs={"class": "RaceData02"}).find_all("span")[4].text      # クラス
            + " "
            + soup.find("div", attrs={"class": "RaceData02"}).find_all("span")[5].text      # 種別
            + " "
            + soup.find("div", attrs={"class": "RaceData02"}).find_all("span")[6].text      # 斤量
            + " "
            + soup.find("div", attrs={"class": "RaceData02"}).find_all("span")[7].text      # 頭数
        )
        info = re.findall(r'\w+', texts)
        # Aコースなどの表記を消去
        info = [s for s in info if s != 'A' and s != 'B' and s != 'C']
        horse_numbers = int(re.findall(r'\d+', info[-1])[0])

        df_info = pd.DataFrame()
        for text in info:
            if text in ["新馬", "未勝利", "1勝クラス","2勝クラス","3勝クラス","オープン", "１勝クラス","２勝クラス","３勝クラス"]:
                df_info["class"] = text
            if "芝" in text:
                df_info["race_type"] = ["芝"]
            if "ダ" in text:
                df_info["race_type"] = ["ダート"]
            if "障" in text:
                df_info["race_type"] = ["障害"]
            if "m" in text:
                df_info["course_len"] = [int(re.findall(r"\d+", text)[-1])]   
            if text in ["良", "重"]:
                df_info["ground_state"] = [text]
            if text in ["稍重","稍"]:
                df_info["ground_state"] = ["稍重"]
            if text in ["不良", "不"]:
                df_info["ground_state"] = ["不良"]
            if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                df_info["weather"] = [text]
        
        # 厩舎名と所属を分離
        local = []
        for i in range(len(df)):
            # print(str(df.at[i,"厩舎"]))
            if "美浦" in str(df.at[i,"厩舎"]):
                local.append("美浦")
                temp = str(df.at[i,"厩舎"])
                temp = temp.replace('美浦', '')
                df.at[i,"厩舎"] = temp
            elif "栗東" in str(df.at[i,"厩舎"]):
                local.append("栗東")
                temp = str(df.at[i,"厩舎"])
                temp = temp.replace('栗東', '')
                df.at[i,"厩舎"] = temp
            else:
                local.append(" ")
        df["所属"] = local

        #馬ID、騎手IDをスクレイピング
        horse_id_list = []
        jockey_id_list = []
        horse_list = soup.find_all("tr", attrs={"class": "HorseList"})
        horse_count = 0
        for horse_infos in horse_list:
            horse_info = horse_infos.find("span", attrs={"class": "HorseName"}).find("a", attrs = {"href" : re.compile("https")})
            horse_id = re.findall(r"\d+", horse_info["href"])
            horse_id_list.append(horse_id[0])

            jockey_info = horse_infos.find("td", attrs={"class": "Jockey"}).find("a", attrs = {"href" : re.compile("https")})
            jockey_id = re.findall(r"\d+", jockey_info["href"])
            jockey_id_list.append(jockey_id[0])
            horse_count += 1
            if(horse_count >=horse_numbers):
                break

        df["horse_id"] = horse_id_list
        df["jockey_id"] = jockey_id_list

        #インデックスをrace_idにする
        df.index = [race_id] * len(df)
        # print(df)
        return info, df_info, df
    except Exception as e:
        scraping_error(e)
        return info, pd.DataFrame(), pd.DataFrame()

def scrape_horse_results(horse_id):
    """馬の過去成績を取得
        Args:
            horse_id(int) : horse_id
        Returns:
            df(pd.DataFrame) : 過去成績
    """
    try:
        url = 'https://db.netkeiba.com/horse/' + horse_id
        # URL存在チェック
        if not url_exists(url):
            print("scrape_horse_results: URL not found, skip", url)
            return pd.DataFrame()
        # スクレイピング
        html = requests.get(url, headers=scraping_header)
        html.encoding = "EUC-JP"
        # soupチェック（pd.read_html前）
        soup = BeautifulSoup(html.text, "html.parser")
        soup = BeautifulSoup(html.content.decode("euc-jp", "ignore"), "html.parser")
        if not validate_soup(soup, url, 'scrape_horse_results', require_table=True):
            return pd.DataFrame()
        # 新馬戦の場合を除外
        if len(pd.read_html(html.text)) < 4:
            return pd.DataFrame()
        df = pd.read_html(html.text)[3]
        
        #受賞歴がある馬の場合、3番目に受賞歴テーブルが来るため、4番目のデータを取得する
        if df.columns[0]=='受賞歴':
            df = pd.read_html(html.text)[4]

        df = df.drop(columns = df.columns[5])
        df = df.drop(columns = df.columns[16 - 1])
        df = df.drop(columns = df.columns[19 - 2])
        df = df.drop(columns = df.columns[24 - 3])
        df = df.drop(columns = df.columns[25 - 4])
        df = df.drop(columns = df.columns[27 - 5])
        df.index = [horse_id] * len(df)
        return df
    
    except Exception as e:
        scraping_error(e)
        return pd.DataFrame()

def scrape_peds(horse_id):
    """血統情報の取得
        Args:
            horse_id(int) : horse_id
        Returns:
            df(pd.DataFrame) : 血統情報
    """
    try:
        url = "https://db.netkeiba.com/horse/ped/" + horse_id
        # スクレイピング
        html = requests.get(url, headers=scraping_header)
        html.encoding = "EUC-JP"

        df = pd.read_html(html.text)[0]

        #重複を削除して1列のSeries型データに直す
        generations = {}
        for i in reversed(range(5)):
            generations[i] = df[i]
            df.drop([i], axis=1, inplace=True)
            df = df.drop_duplicates()
        ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)
        for i in ped.index:
            # 生年以降を消去
            pattern =  re.findall(r'\d+', ped.iat[i]) 
            if pattern:
                pos = str(ped.iat[i]).find(pattern[0])
                temp = str(ped.iat[i][:pos])
                ped.iat[i] = temp
        ped = ped.reset_index(drop=True)
        ped = ped.str.lstrip()

        return ped
    
    except Exception as e:
        scraping_error(e)
        return pd.DataFrame()


if __name__ == "__main__":
    scrape_race_card("202403030401")