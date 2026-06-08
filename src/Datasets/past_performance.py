import os
import re
import sys

from datetime import date, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

# pycache を生成しない
sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import get_race_id
import name_header
import scraping
import analysis_race_info
import race_results

def past_performance_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def make_past_performance_dataset(horse_id):
    """ hotse_idから、過去成績のDataFrameを作成 
        Args:
            horse_id (str) : horse_id
        Returns:
            DataFrame: horse_idの過去成績をDataFrame型で返す    
    """
    url = 'https://db.netkeiba.com/horse/' + horse_id
    # スクレイピング
    horse_results_df = scraping.scrape_df(url)
    # print(horse_results_df)
    try:
        # 新馬戦の場合を除外
        if len(horse_results_df) < 4:
            return pd.DataFrame()
        
        # 過去のレース結果テーブルを取得
        df = horse_results_df[3]
        
        #受賞歴がある馬の場合、3番目に受賞歴テーブルが来るため、4番目のデータを取得する
        if df.columns[0]=='受賞歴':
            df = horse_results_df[4]

        # 不要な要素を消去する
        df = df.drop(columns = df.columns[5])
        df = df.drop(columns = df.columns[16 - 1])
        df = df.drop(columns = df.columns[19 - 2])
        df = df.drop(columns = df.columns[24 - 3])
        df = df.drop(columns = df.columns[25 - 4])
        df = df.drop(columns = df.columns[27 - 5])
        df = df.reset_index(drop = True)

        return df
    except Exception as e:
        print("error id:", horse_id)
        past_performance_error(e)
        return pd.DataFrame

def save_past_performance_dataset(horse_id, past_performance_df):
    """ past_performanceのDataFrameを保存 
        Args:
            horse_id (int) :horse_id
            past_performance_df(pd.DataFrame） : past_performanceのデータセット
    """
    try:
        if any(past_performance_df):
            # csv/pickleに保存
            past_performance_df.to_csv(name_header.DATA_PATH + "PastPerformance\\" + str(horse_id) + '.csv')
            past_performance_df.to_pickle(name_header.DATA_PATH + "PastPerformance\\" + str(horse_id) + '.pickle')
    except Exception as e:
            past_performance_error(e)

def get_past_performance_dataset(horse_id):
    """ past_performanceのデータセットを取得 
        Args:
            horse_id (int) :horse_id
    """

    # csvを読み込む 
    path = name_header.DATA_PATH + "PastPerformance\\" + str(horse_id) + '.csv'
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col = 0, dtype = str)
    else :
        df = pd.DataFrame()

    return df

def get_horse_id_from_peds_dataset(place_id, year):
    """  過去の結果からhorse_idを抽出
    Args:
        place_id (int) : 開催コースid
        year(int) : 開催年
    Returns:
        list(str): horse_idのリスト    
    """
    # 過去の血統データを取得する
    path = name_header.DATA_PATH + name_header.PLACE_LIST[place_id - 1] + "/" + str(year) + "_peds.csv"
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col = 0)
    else :
        print("not_is_file:" + path)
        return []

    # horse_idのみを抽出し返す
    return df.index

def get_horse_id_list_from_race_id_list(race_id_list):
    """ race_id_listから出走馬のhorse_idを取得
        Args:
            race_id(int) : race_id  
        Returns:
            horse_id_list : horse_idのリスト
    """ 
    try:
        horse_id_list = []
        for race_id in race_id_list:
            horse_id_list.append(get_horse_id_from_race_id(race_id))
        horse_id_list = sum(horse_id_list, [])
        return horse_id_list
    except Exception as e:
            past_performance_error(e)
            return horse_id_list

def get_horse_id_from_race_id(race_id):
    """ race_idtから出走馬のhorse_idを取得
        Args:
            race_id(int) : race_id  
        Returns:
            horse_id_list : horse_idのリスト
    """ 
    horse_id_list = []
    # race_idから年,place_idを抽出
    year = str(race_id)[0] + str(race_id)[1] + str(race_id)[2] + str(race_id)[3] 
    place_id = int(str(race_id)[4] + str(race_id)[5])

    # レース結果を取得
    df = race_results.get_race_results_csv(place_id, year)
    if df.empty:
        return horse_id_list

    # race_idのレースのみを抽出
    df.index = df.index.astype(str)
    df = df[race_id:race_id]
    # horse_idのみを抽出
    horse_id_list = df.loc[:,"horse_id"].to_list()

    return horse_id_list

def get_past_race_id(horse_result):
    """ 過去の成績をrace_id_listに変換
        Args:
            horse_result(pd.DataFrame) : 過去成績のデータセット  
        Returns:
            race_id_list : race_idのリスト
    """
    if horse_result.empty:
        race_id_list = []
    else :
        race_id_list = horse_result["race_id"].tolist()
    
    return race_id_list

def get_past_race_info(horse_id, race_id, race_num):
    """当該レースより過去の指定レース数取得
        Args:
            horse_id(int) : horse_id
            race_id(int) : race_id
            race_num(int) : 抽出するレース数
        Returns:
            horse_result : 指定レース数の過去レース結果
    """
    try: 
        # データセットの取得
        horse_result = get_past_performance_dataset(horse_id)
        # データセットがない場合は新規作成
        if horse_result.empty:
            horse_results_df = make_past_performance_dataset_from_race_results(str(horse_id))
            save_past_performance_dataset(str(horse_id), horse_results_df)
            horse_result = get_past_performance_dataset(horse_id)

        # 当該レースより過去のレースを取得
        horse_result = reset_horse_result(horse_result, race_id)
        
        # 指定レース数取得
        if len(horse_result.index) > race_num:
            horse_result = horse_result[0:race_num]

        return horse_result
    except Exception as e:
        past_performance_error(e)
        return pd.DataFrame()

def reset_horse_result(horse_result, race_id):
    """過去の成績のうちrace_id以降のレースを消去
        Args:
            horse_result(pd.DataFrame) : 過去のレース結果
            race_id(int) : race_id
        Returns:
            horse_result : race_id以降の過去レース結果
    """
    race_id_list = get_past_race_id(horse_result.reset_index())
    
    # race_id　と一致する場所を取得
    idx = -1
    for i in range(len(race_id_list)):
        if str(race_id) == race_id_list[i]:
            idx = i
            break
    
    # race_id　以降を消去
    new_horse_results = horse_result
    if idx > 0:
        if idx == len(race_id_list):
            new_horse_results = pd.DataFrame()
        else:
            new_horse_results = horse_result[idx + 1:len(race_id_list)]
    
    return new_horse_results.reset_index(drop = True)

def make_past_performanece_datasets(horse_id_list):
    """ horse_id_listから過去成績のデータセットを作成  
        Args:
            horse_id_list : horse_idのリスト
    """ 
    for horse_id in tqdm(horse_id_list):
        # データセットの取得
        horse_results_df = make_past_performance_dataset_from_race_results(str(horse_id))
        # print(horse_results_df)
        save_past_performance_dataset(str(horse_id), horse_results_df)

def make_past_performance_dataset_from_race_results(horse_id):
    """全race_resultsから horse_id の過去成績を past_performance 形式で作成"""
    all_race_df = []
    base_dir = name_header.DATA_PATH + "RaceResults"

    # --- RaceResults全体を走査して全レース情報を収集 ---
    for place in sorted(os.listdir(base_dir)):
        place_path = os.path.join(base_dir, place)
        if not os.path.isdir(place_path):
            continue

        for file in sorted(os.listdir(place_path)):
            if not file.endswith("_race_results.csv"):
                continue
            csv_path = os.path.join(place_path, file)
            try:
                df = pd.read_csv(csv_path, dtype=str)
                if "horse_id" not in df.columns:
                    continue
                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "race_id"})
                df["place_id"] = place  # 開催場
                all_race_df.append(df)
            except Exception as e:
                print(f"⚠️ {csv_path} 読み込みエラー: {e}")

    if not all_race_df:
        print(f"❌ RaceResultsが見つかりません。")
        return pd.DataFrame()

    full_df = pd.concat(all_race_df, ignore_index=True)
    full_df = full_df.fillna("")

    # 対象馬のみ抽出
    df = full_df[full_df["horse_id"] == str(horse_id)].copy()
    if df.empty:
        print(f"⚠ horse_id={horse_id} のデータが見つかりません。")
        return pd.DataFrame()

    # --- race_idごとに他馬データを参照して派生情報を生成 ---
    records = []
    for _, row in df.iterrows():
        race_id = row["race_id"]
        same_race = full_df[full_df["race_id"] == race_id].copy()

        # 除外・取消は頭数から除外（中止は残す）
        def valid_starter(r):
            txt = str(r["着順"])
            if any(w in txt for w in ["除", "取", "取消"]):
                return False
            return True

        valid_starters = same_race[same_race.apply(valid_starter, axis=1)]
        headcount = len(valid_starters)

        # 勝ち馬・2着馬
        first = valid_starters[valid_starters["着順"] == "1"]
        second = valid_starters[valid_starters["着順"] == "2"]
        first_name = first.iloc[0]["馬名"] if not first.empty else ""
        second_name = second.iloc[0]["馬名"] if not second.empty else ""

        # タイム差計算
        def to_seconds(t):
            if not t or t in ["nan", "NaN", ""]:
                return np.nan
            t = t.replace("0:", "") if t.startswith("0:") else t
            parts = t.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(parts[-1])

        # --- タイム差計算処理 ---
        if any(x in str(row.get("着順", "")) for x in ["除", "取", "中", "失"]) or  not row.get("タイム"):
            # 中止レース、またはタイム未記録 → 着差は NaN
            margin = np.nan
        else:
            this_time = to_seconds(row.get("タイム", ""))
            first_time = to_seconds(first.iloc[0]["タイム"]) if not first.empty else np.nan
            second_time = to_seconds(second.iloc[0]["タイム"]) if not second.empty else np.nan

            margin = ""
            if not np.isnan(this_time) and not np.isnan(first_time):
                diff = round(this_time - first_time, 1)
                if str(row["着順"]) == "1":
                    # 1着の場合：2着との差をマイナス値で表示
                    if not np.isnan(second_time):
                        diff_to_second = round(second_time - this_time, 1)
                        margin = f"-{diff_to_second:.1f}" if diff_to_second > 0 else "-0.0"
                    else:
                        margin = "-0.0"
                else:
                    margin = f"{diff:.1f}" if diff >= 0 else ""

        # 勝ち馬欄
        if str(row["着順"]) == "1":
            winner_text = f"({second_name})" if second_name else ""
        else:
            winner_text = first_name

        # --- 統一フォーマット化 ---
        record = {
            "race_id": race_id,
            "日付": row["date"].replace("年", "/").replace("月", "/").replace("日", ""),
            "開催": name_header.NAME_LIST[int(row["place_id"][:2]) - 1] if str(row["place_id"][:2]).isdigit() else "",
            "天気": row.get("weather", ""),
            "R": str(race_id)[-2:],
            "レース名": "",  # 後でrace_name_mapで補完
            "class": row.get("class", ""),
            "頭数": headcount,
            "枠番": row.get("枠番", ""),
            "馬番": row.get("馬番", ""),
            "オッズ": row.get("単勝", ""),
            "人気": row.get("人気", ""),
            "着順": row.get("着順", ""),
            "騎手": row.get("騎手", ""),
            "斤量": row.get("斤量", ""),
            "race_type": row.get("race_type", ""),
            "course_len": row.get("course_len", ""),
            "ground_state": row.get("ground_state", ""),
            "タイム": row.get("タイム", ""),
            "着差": margin,
            "通過": row.get("通過", ""),
            "上り": row.get("上り", ""),
            "馬体重": row.get("馬体重", ""),
            "勝ち馬 (2着馬)": winner_text,
        }
        records.append(record)

    result_df = pd.DataFrame(records)

    # --- レース名補完 ---
    race_name_map = {}
    race_time_dir = os.path.join(name_header.TEXT_PATH, "race_calendar/race_time_id_list")
    for file in os.listdir(race_time_dir):
        if not file.endswith(".csv"):
            continue
        try:
            df_info = pd.read_csv(os.path.join(race_time_dir, file), dtype=str)
            for _, r in df_info.iterrows():
                race_name_map[str(r["race_id"])] = str(r.get("race_name", ""))
        except Exception:
            continue
    result_df["レース名"] = result_df["race_id"].map(race_name_map).fillna("")

    # --- 並び順統一 ---
    out_cols = [
        "race_id", "日付", "開催", "天気", "R", "レース名", "class", "頭数",
        "枠番", "馬番", "オッズ", "人気", "着順", "騎手", "斤量",
        "race_type", "course_len", "ground_state", "タイム", "着差",
        "通過", "上り", "馬体重", "勝ち馬 (2着馬)"
    ]
    for c in out_cols:
        if c not in result_df.columns:
            result_df[c] = ""

    result_df = result_df[out_cols]
    # 重複レースの削除
    result_df = result_df.drop_duplicates(subset="race_id", keep="last")
    # --- ソート（日付新→古） ---
    try:
        result_df["日付_dt"] = pd.to_datetime(result_df["日付"], errors="coerce")
        result_df = result_df.sort_values("日付_dt", ascending=False).drop(columns=["日付_dt"])
    except Exception:
        pass

    # print(f"✅ horse_id={horse_id} の過去成績 {len(result_df)}件を整形しました。")
    return result_df.reset_index(drop=True)

def normalize_past_performance_format(df_old: pd.DataFrame) -> pd.DataFrame:
    """
    旧フォーマットのpast_performanceを新フォーマットに変換する
    """
    
    if df_old.empty:
        return pd.DataFrame()
    df = df_old.copy()

    # ✅ 新フォーマット判定
    new_columns = {
        "race_id", "日付", "開催", "天気", "R", "レース名", "class", "頭数",
        "枠番", "馬番", "オッズ", "人気", "着順", "騎手", "斤量",
        "race_type", "course_len", "ground_state", "タイム", "着差",
        "通過", "上り", "馬体重", "勝ち馬 (2着馬)"
    }
    if new_columns.issubset(set(df_old.columns)):
        # すでに新形式 → 軽微な整形のみ
        
        df["日付"] = pd.to_datetime(df["日付"], errors="coerce").dt.strftime("%Y/%m/%d")
        if "class" in df.columns:
            df["class"] = df["class"].apply(normalize_class_text)
        if "レース名" in df.columns:
            df["レース名"] = df["レース名"].apply(clean_race_name)
        return df

    # --- 列名を標準化 ---
    rename_map = {
        "天 気": "天気",
        "枠 番": "枠番",
        "馬 番": "馬番",
        "頭 数": "頭数",
        "オ ッ ズ": "オッズ",
        "人 気": "人気",
        "着 順": "着順",
        "斤 量": "斤量",
        "馬 場": "ground_state",
    }
    df = df.rename(columns=rename_map)

    # --- 不要な列削除 ---
    drop_cols = ["ペース"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # レースクラス判定関数
    def extract_class(race_name):
        if pd.isna(race_name):
            return ""
        name = str(race_name)

        # 半角全角を統一して大文字化
        name = name.upper().replace("Ｇ", "G")

        # パターンマップ（順序が重要：上位カテゴリを先に）
        patterns = [
            (r"G[ⅠI1]", "オープン"),
            (r"G[ⅡI2]", "オープン"),
            (r"G[ⅢI3]", "オープン"),
            (r"OP|OPEN|オープン|ｵｰﾌﾟﾝ", "オープン"),
            (r"3勝クラス|３勝|三勝", "3勝クラス"),
            (r"2勝クラス|２勝|二勝", "2勝クラス"),
            (r"1勝クラス|１勝|一勝", "1勝クラス"),
            (r"未勝利", "未勝利"),
            (r"新馬", "新馬"),
        ]

        for pat, label in patterns:
            if re.search(pat, name):
                return label
        return ""  # 不明な場合は空白
    
    # --- class列を追加 ---
    if "class" not in df.columns:
        df["class"] = df["レース名"].apply(extract_class)


    # --- 距離列を分割（芝2200 → race_type="芝", course_len="2200"） ---
    df["race_type"] = df["距離"].astype(str).str.extract(r"([芝ダ障])")[0].fillna("")
    df["course_len"] = df["距離"].astype(str).str.extract(r"(\d+)")[0].fillna("")

    # --- 馬体重整形 ---
    df["馬体重"] = df["馬体重"].astype(str).str.extract(r"(\d+)")[0].fillna("")

    # --- オッズ, 人気, 枠番, 馬番などの整形 ---
    for col in ["オッズ", "人気", "枠番", "馬番", "斤量"]:
        df[col] = df[col].astype(str).str.replace(",", "").str.strip()
        df[col] = df[col].replace("", None)

    # --- 着差を数値化 ---
    def parse_margin(x):
        try:
            s = str(x).replace("−", "-").replace("+", "").strip()
            return float(s)
        except:
            return ""
    df["着差"] = df["着差"].apply(parse_margin)

    # --- 日付整形 ---
    df["日付"] = (
        df["日付"].astype(str)
        .str.replace("年", "/", regex=False)
        .str.replace("月", "/", regex=False)
        .str.replace("日", "", regex=False)
        .str.strip()
    )

    # --- race_id生成 ---
    place_map = {
        "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06",
        "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"
    }
    def make_race_id(row):
        # 年
        year = str(pd.to_datetime(row["日付"]).year)
        # 開催の左数字（例: 3中京1 → 3）
        m = re.match(r"(\d+)([^\d]+)(\d+)", str(row["開催"]))
        if not m:
            return None
        left_num, course_name, right_num = m.groups()
        # 開催地コード
        course_code = place_map.get(course_name, "00")
        # Rを2桁に
        r_num = f"{int(float(row['R'])):02d}"
        # race_id組み立て
        return f"{year}{course_code}{int(left_num):02d}{int(right_num):02d}{r_num}"
    
    df["race_id"] = df.apply(make_race_id, axis=1)

    # --- 開催名を簡略化（例: "3中京1"→"中京"） ---
    df["開催"] = df["開催"].astype(str).str.replace(r"^\d+", "", regex=True).str.replace(r"\d+$", "", regex=True).str.strip()

    # --- カラム整列 ---
    expected_cols = [
        "race_id", "日付", "開催", "天気", "R", "レース名", "class", "頭数", "枠番", "馬番",
        "オッズ", "人気", "着順", "騎手", "斤量", "race_type", "course_len",
        "ground_state", "タイム", "着差", "通過", "上り", "馬体重", "勝ち馬 (2着馬)"
    ]
    df = df[[c for c in expected_cols if c in df.columns]]

    # --- ソート（日付降順） ---
    try:
        df["日付_dt"] = pd.to_datetime(df["日付"], errors="coerce")
        df = df.sort_values("日付_dt", ascending=False).drop(columns=["日付_dt"])
    except Exception:
        pass

    print(f"✅ {len(df)}件の旧データを新フォーマットに変換しました。")
    return df

def normalize_class_text(text):
    """
    クラス表記を正規化する。
    例: '３勝クラス' → '3勝クラス'
    """
    if pd.isna(text):
        return text
    # 全角→半角変換（数字・英字）
    text = text.translate(str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    ))
    # よくあるブレを修正
    text = re.sub(r'\s+', '', text)  # スペース削除
    text = re.sub(r'勝ｸﾗｽ', '勝クラス', text)
    text = re.sub(r'未勝利', '未勝利', text)
    text = re.sub(r'新馬', '新馬', text)
    text = re.sub(r'ｸﾗｽ', 'クラス', text)
    text = re.sub(r'ｵｰﾌﾟﾝ', 'オープン', text)
    return text

def clean_race_name(text):
    """
    レース名に含まれる（3勝クラス）などの表記を削除する。
    例: 'ムーンライトH(3勝クラス)' → 'ムーンライトH'
    """
    if pd.isna(text):
        return text
    # ()内のクラス・レベル表記を削除
    text = re.sub(r'（.*?クラス.*?）', '', text)  # 全角括弧対応
    text = re.sub(r'\(.*?クラス.*?\)', '', text)  # 半角括弧対応
    text = re.sub(r'\s+$', '', text)  # 末尾の空白を削除
    return text

def update_past_performance(horse_id):
    # --- 既存の past_performance 読み込み ---
    past_performance_root = name_header.DATA_PATH + "PastPerformance"
    past_path = os.path.join(past_performance_root, f"{horse_id}.csv")
    if os.path.exists(past_path):
        past_df = pd.read_csv(past_path, dtype=str)
    else:
        past_df = pd.DataFrame()
        print("⚠️ 過去データなし（新規作成）")

    print("processing:", horse_id)
    # --- 🧩 両方を共通フォーマットに揃える ---
    existing_df_norm = normalize_past_performance_format(past_df)
    # print(existing_df_norm)
   

    # --- 追加データをマージ ---
    add_df = make_past_performance_dataset_from_race_results(horse_id)
    # print(add_df)
    if not add_df.empty:
        updated_df = pd.concat([existing_df_norm, add_df], ignore_index=True)
        updated_df = updated_df.drop_duplicates(subset=["race_id"], keep="first")
    else:
        updated_df = existing_df_norm
        print("✅ 新しいレースはありません")

    # --- 日付順に並べ替え ---
    if "日付" in updated_df.columns:
        updated_df["日付"] = pd.to_datetime(updated_df["日付"], errors="coerce")
        updated_df = updated_df.sort_values("日付", ascending=False).reset_index(drop=True)
    
    # 日付を正規化（例：2025-09-07 → 2025/09/07）
    updated_df["日付"] = pd.to_datetime(updated_df["日付"], errors="coerce").dt.strftime("%Y/%m/%d")

    # クラス表記を正規化
    updated_df["class"] = updated_df["class"].apply(normalize_class_text)

    # レース名のクラス表記を削除
    if "レース名" in updated_df.columns:
        updated_df["レース名"] = updated_df["レース名"].apply(clean_race_name)

    # ===== 列の並び統一 =====
    expected_cols = [
        "race_id", "日付", "開催", "天気", "R", "レース名", "class", "頭数",
        "枠番", "馬番", "オッズ", "人気", "着順", "騎手", "斤量",
        "race_type", "course_len", "ground_state", "タイム", "着差",
        "通過", "上り", "馬体重", "勝ち馬 (2着馬)"
    ]
    # 存在しない列を補完（NaNで）
    for col in expected_cols:
        if col not in updated_df.columns:
            updated_df[col] = pd.NA
    updated_df = updated_df[expected_cols]

    # --- 保存 ---
    os.makedirs(past_performance_root, exist_ok=True)
    updated_df.to_csv(past_path, index=False, encoding="utf-8-sig")
    print(f"💾 保存完了: {past_path} （{len(updated_df)} 件）")

    return updated_df

def make_past_performance_dataset_from_race_id_list(race_id_list):
    """ race_id_listから過去成績のデータセットを作成  
        Args:
            race_id_list : race_idのリスト
    """ 
    horse_id_list = get_horse_id_list_from_race_id_list(race_id_list)
    if any(horse_id_list):
        make_past_performanece_datasets(horse_id_list)

def weekly_update_past_performance(day = date.today()):
    """ 指定した日にちから、１週間分のデータセットを更新  
        Args:
            day(Date) : 日（初期値：今日）
    """ 
    for place_id in tqdm(range(1, len(name_header.PLACE_LIST) + 1)):
        print("[WeeklyUpdate]", name_header.PLACE_LIST[place_id -1] , "PastPerformance")
        race_id_list = get_race_id.get_past_weekly_id(place_id, day)
        make_past_performance_dataset_from_race_id_list(race_id_list)
        
def monthly_update_past_performance(day = date.today()):
    """ 指定した日にちまでのその年のデータセットを更新  
        Args:
            day(Date) : 日（初期値：今日）
    """ 
    for place_id in tqdm(range(1, len(name_header.PLACE_LIST) + 1)):
        print("[MonthlyUpdate]", name_header.PLACE_LIST[place_id -1],  "PastPerformance")
        race_id_list = get_race_id.get_past_year_id(place_id, day)
        make_past_performance_dataset_from_race_id_list(race_id_list)

def make_all_past_performance(year = date.today().year):
    """ 指定した年までの、すべてのデータセットを作成 
        Args:
            day(Date) : 日（初期値：今日）
    """ 
    for y in range(2019, year + 1):
        for place_id in range(1, len(name_header.PLACE_LIST) + 1):
            print("[NewMake]" + str(y) + ":" + name_header.PLACE_LIST[place_id -1] + " PastPerformance")
            race_id_list = get_race_id.get_year_id_all(place_id,y)
            make_past_performance_dataset_from_race_id_list(race_id_list)

if __name__ == "__main__":
    monthly_update_past_performance()
