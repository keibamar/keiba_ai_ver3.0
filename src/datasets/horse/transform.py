"""horse（血統・過去成績）データセットの変換ロジック（純粋関数のみ）

旧 src/legacy_datasets/horse_peds.py / peds_results.py / past_performance.py の
純粋関数部分を移植したもの。
"""

import re

import numpy as np
import pandas as pd

from src.datasets.horse import model


def normalize_horse_name_strings(s):
    """馬名文字列を正規化する（カタカナ・英語表記の余分な部分を除去）

    Args:
        s (str): 馬名文字列

    Returns:
        str: 正規化した馬名文字列
    """
    if not isinstance(s, str):
        s = str(s)

    # カタカナで始まる場合（全角カタカナ＋長音記号＋中点など）
    if re.match(r"^[ァ-ヶー・ヴ]", s):
        # 先頭のカタカナ部分だけ抽出
        m = re.match(r"^([ァ-ヶー・ヴ]+)", s)
        if m:
            s = m.group(1)
    # 英語で始まる場合（A-Z/a-z）
    elif re.match(r"^[A-Za-z]", s):
        # 括弧と中身を削除
        s = re.sub(r"\([^)]*\)", "", s)

    # 行末の空白を削除
    s = s.strip()
    return s


def delete_invalid_strings(peds_df):
    """血統データの各要素から生年以降の文字列を消去する

    Args:
        peds_df (pd.DataFrame): 血統データのDataFrame

    Returns:
        pd.DataFrame: 不要な文字列を消去した血統データのDataFrame
    """
    for l in range(len(peds_df)):
        pattern = re.findall(r"\d+", peds_df.iloc[l])
        if pattern:
            # 生年以降を消去
            pos = str(peds_df.iloc[l]).find(pattern[0])
            temp = str(peds_df.iloc[l][:pos])
            peds_df.iloc[l] = temp
        # 前後の空白を消去
        peds_df.iloc[l] = str(peds_df.iloc[l]).strip()

    return peds_df


def to_seconds(t):
    """タイム文字列を秒数に変換する

    Args:
        t (str): "1:23.4" や "23.4" のようなタイム文字列

    Returns:
        float: 秒数（変換できない場合はnp.nan）
    """
    if not t or t in ["nan", "NaN", ""]:
        return np.nan
    t = t.replace("0:", "") if t.startswith("0:") else t
    parts = t.split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[-1])


def get_past_race_id(horse_result):
    """過去の成績をrace_id_listに変換する

    Args:
        horse_result (pd.DataFrame): 過去成績のデータセット

    Returns:
        list: race_idのリスト
    """
    if horse_result.empty:
        return []
    return horse_result["race_id"].tolist()


def reset_horse_result(horse_result, race_id):
    """過去の成績のうちrace_id以降のレースを消去する

    Args:
        horse_result (pd.DataFrame): 過去のレース結果
        race_id (str): race_id

    Returns:
        pd.DataFrame: race_id以降を除いた過去レース結果
    """
    race_id_list = get_past_race_id(horse_result.reset_index())

    # race_idと一致する場所を取得
    idx = -1
    for i in range(len(race_id_list)):
        if str(race_id) == race_id_list[i]:
            idx = i
            break

    # race_id以降を消去
    new_horse_results = horse_result
    if idx > 0:
        if idx == len(race_id_list):
            new_horse_results = pd.DataFrame()
        else:
            new_horse_results = horse_result[idx + 1 : len(race_id_list)]

    return new_horse_results.reset_index(drop=True)


def normalize_class_text(text):
    """クラス表記を正規化する（例: '３勝クラス' -> '3勝クラス'）

    Args:
        text (str): クラス表記文字列

    Returns:
        str: 正規化したクラス表記文字列
    """
    if pd.isna(text):
        return text
    # 全角→半角変換（数字・英字）
    text = text.translate(str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    ))
    # よくあるブレを修正
    text = re.sub(r"\s+", "", text)  # スペース削除
    text = re.sub(r"勝ｸﾗｽ", "勝クラス", text)
    text = re.sub(r"未勝利", "未勝利", text)
    text = re.sub(r"新馬", "新馬", text)
    text = re.sub(r"ｸﾗｽ", "クラス", text)
    text = re.sub(r"ｵｰﾌﾟﾝ", "オープン", text)
    return text


def clean_race_name(text):
    """レース名に含まれる「(3勝クラス)」などの表記を削除する

    Args:
        text (str): レース名文字列

    Returns:
        str: クラス表記を除いたレース名文字列
    """
    if pd.isna(text):
        return text
    # ()内のクラス・レベル表記を削除
    text = re.sub(r"（.*?クラス.*?）", "", text)  # 全角括弧対応
    text = re.sub(r"\(.*?クラス.*?\)", "", text)  # 半角括弧対応
    text = re.sub(r"\s+$", "", text)  # 末尾の空白を削除
    return text


def normalize_past_performance_format(df_old):
    """旧フォーマットのpast_performanceを新フォーマットに変換する

    Args:
        df_old (pd.DataFrame): 旧フォーマットのpast_performance

    Returns:
        pd.DataFrame: 新フォーマットのpast_performance
    """
    if df_old.empty:
        return pd.DataFrame()
    df = df_old.copy()

    # 新フォーマット判定
    new_columns = set(model.PAST_PERFORMANCE_COLUMNS)
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
        except Exception:
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
    def make_race_id(row):
        # 年
        year = str(pd.to_datetime(row["日付"]).year)
        # 開催の左数字（例: 3中京1 → 3）
        m = re.match(r"(\d+)([^\d]+)(\d+)", str(row["開催"]))
        if not m:
            return None
        left_num, course_name, right_num = m.groups()
        # 開催地コード
        course_code = model.PLACE_MAP.get(course_name, "00")
        # Rを2桁に
        r_num = f"{int(float(row['R'])):02d}"
        # race_id組み立て
        return f"{year}{course_code}{int(left_num):02d}{int(right_num):02d}{r_num}"

    df["race_id"] = df.apply(make_race_id, axis=1)

    # --- 開催名を簡略化（例: "3中京1"→"中京"） ---
    df["開催"] = df["開催"].astype(str).str.replace(r"^\d+", "", regex=True).str.replace(r"\d+$", "", regex=True).str.strip()

    # --- カラム整列 ---
    df = df[[c for c in model.PAST_PERFORMANCE_COLUMNS if c in df.columns]]

    # --- ソート（日付降順） ---
    try:
        df["日付_dt"] = pd.to_datetime(df["日付"], errors="coerce")
        df = df.sort_values("日付_dt", ascending=False).drop(columns=["日付_dt"])
    except Exception:
        pass

    return df


def calc_peds_placed_rate(peds_data):
    """データセットから着度数を計算する

    Args:
        peds_data (pd.DataFrame): 血統毎のレース結果データセット

    Returns:
        list: [1着数, 2着数, 3着数, 着外数]
    """
    return [
        peds_data[peds_data["着順"] == "1"]["着順"].count(),
        peds_data[peds_data["着順"] == "2"]["着順"].count(),
        peds_data[peds_data["着順"] == "3"]["着順"].count(),
        len(peds_data) - (
            peds_data[peds_data["着順"] == "1"]["着順"].count()
            + peds_data[peds_data["着順"] == "2"]["着順"].count()
            + peds_data[peds_data["着順"] == "3"]["着順"].count()
        ),
    ]


def calc_peds_data(df_result, course_len, race_class):
    """血統の着度数を計算する

    Args:
        df_result (pd.DataFrame): 血統毎のレース結果データセット
        course_len (str): コース距離
        race_class (str): クラス

    Returns:
        pd.DataFrame: 血統着度数データセット（place/course/classの3列）
    """
    return_data = []
    # 同競馬場スコア
    return_data.append(calc_peds_placed_rate(df_result))

    # 同コース
    df_result = df_result[df_result["course_len"] == str(course_len)]
    return_data.append(calc_peds_placed_rate(df_result))

    # 同条件
    df_result = df_result[df_result["class"] == str(race_class)]
    return_data.append(calc_peds_placed_rate(df_result))

    return pd.DataFrame(zip(*return_data), columns=["place", "course", "class"])


def get_race_type_data(df_result, race_type, ground_state):
    """df_resultのレースタイプと馬場状態を抽出する

    Args:
        df_result (pd.DataFrame): race_resultデータセット
        race_type (str): コースタイプ
        ground_state (str): 馬場状態

    Returns:
        pd.DataFrame: 抽出したrace_result
    """
    df_result = df_result[df_result["race_type"] == race_type]
    df_result = df_result[df_result["ground_state"] == ground_state]

    return df_result


def output_results(df_peds_results):
    """血統ごとに着順を集計してDataFrameを返す

    Args:
        df_peds_results (pd.DataFrame): 血統別レース結果データセット

    Returns:
        pd.DataFrame: 血統ごとの着度数集計結果
    """
    if df_peds_results.empty:
        return pd.DataFrame()

    result_list = []
    for peds, sub_df in df_peds_results.groupby("peds_0"):
        first = (sub_df["着順"] == 1).sum()
        second = (sub_df["着順"] == 2).sum()
        third = (sub_df["着順"] == 3).sum()
        others = ((sub_df["着順"] > 3) & (sub_df["着順"].notna())).sum()

        result_list.append({
            "血統": peds,
            "1着": first,
            "2着": second,
            "3着": third,
            "着外": others,
        })

    result_df = pd.DataFrame(result_list)
    result_df = result_df.sort_values(by=["1着", "2着", "3着"], ascending=False)

    return result_df
