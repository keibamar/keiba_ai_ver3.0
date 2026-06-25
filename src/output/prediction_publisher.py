"""予想結果のテキスト化・メール／X(Twitter)配信を担うモジュール

旧 src/RacePrediction/make_text.py（make_race_text/extract_top5_pred）、
libs/mail_api.py、libs/post_text.py からの移植。

配当結果レポート（make_return_text等）・日次配信オーケストレーションは対象外。
"""

import os
import smtplib
import ssl
from email.mime.text import MIMEText

import tweepy
from dotenv import find_dotenv, load_dotenv

from src.config import paths
from src.config.constants import NAME_LIST
from src.config.lists import SYMBOL_LIST
from src.managers import race_card_dataset_manager

load_dotenv(find_dotenv())

# Gmail送信用（.envのGMAIL_ACCOUNT/GMAIL_APP_PASSWORD/GMAIL_SEND_TOで設定）
GMAIL_ACCOUNT = os.environ["GMAIL_ACCOUNT"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
GMAIL_SEND_TO = os.environ["GMAIL_SEND_TO"]

# X(Twitter) APIキー（.envのX_API_KEY等で設定）
X_API_KEY = os.environ["X_API_KEY"]
X_API_SECRET = os.environ["X_API_SECRET"]
X_ACCESS_TOKEN = os.environ["X_ACCESS_TOKEN"]
X_ACCESS_TOKEN_SECRET = os.environ["X_ACCESS_TOKEN_SECRET"]


def make_test_error(e):
    """エラー時動作を記載する

    Args:
        e (Exception) : エラー内容
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")


def extract_top5_pred(race_data_df):
    """予想結果の上位5頭のリストを返す

    Args:
        race_data_df(pd.DataFrame) : 出馬表データセット
    Returns:
        result_list(list) : 上位5頭の[馬番, 馬名]のリスト(昇順)
    """
    result_list = []
    for i in range(1, 6):
        temp = race_data_df[race_data_df["rank"] == i].reset_index(drop=True)
        if not temp.empty:
            num = temp.at[0, "馬番"]
            name = temp.at[0, "馬名"]
            result_list.append([num, name])
    return result_list


def make_race_text(race_day, race_id):
    """レースの予想のテキスト作成

    Args:
        race_day(date) : レース開催日
        race_id(int) : race_id
    """
    # 予想結果を抽出
    race_data_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
    if "rank" not in race_data_df.columns:
        print("not rank:" + str(race_id))
        return
    try:
        # 予想結果から上位5頭を抽出
        pred_list = extract_top5_pred(race_data_df)

        # テキストファイルの準備
        folder_path = os.path.join(paths.RACE_PREDICTION_TEXT_PATH, race_day.strftime("%Y%m%d"))
        os.makedirs(folder_path, exist_ok=True)
        text_data_path = os.path.join(folder_path, f"{race_id}.txt")

        f = open(text_data_path, "w", encoding="UTF-8")

        # 開催情報の抽出
        place_id = int(str(race_id)[4] + str(race_id)[5])
        race_num = int(str(race_id)[10] + str(race_id)[11])
        time_id_df = race_card_dataset_manager.get_race_time_id_list_df(race_day)
        row = time_id_df[time_id_df["race_id"] == str(race_id)]
        if not row.empty:
            race_name = row.iloc[0]["race_name"]
            race_time = row.iloc[0]["race_time"]
            start_time = race_time[:2] + ":" + race_time[2:]
        else:
            race_name = ""
            start_time = ""

        # 日付の出力
        f.write(str(race_day.year) + "/" + str(race_day.month) + "/" + str(race_day.day) + "\n")
        # 開催情報の出力
        f.write(NAME_LIST[place_id - 1] + str(race_num) + "R" + " " + race_name + " " + start_time + "\n\n")
        # 予想の出力
        for rank in range(5):
            if rank < len(pred_list):
                f.write(" " + SYMBOL_LIST[rank] + " " + str(pred_list[rank][0]) + " " + pred_list[rank][1] + "\n")
        f.write("\n\n")

        # タグの出力
        f.write("#MAR競馬予想\n")
        f.write("#競馬予想AI\n")
        f.write("#競馬 #競馬予想\n")
        f.write("#" + NAME_LIST[place_id - 1] + "競馬場\n")

        f.close()
    except Exception as e:
        make_test_error(e)


# txtファイルの内容を読み込む関数（改行を考慮）
def read_txt_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print("指定されたファイルが見つかりません！")
        return ""


# txtファイルの内容を処理し、タイトルと本文を返す関数
def process_txt_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]  # 空行を削除
            if len(lines) < 2:
                print("ファイルに十分な内容がありません！")
                return "タイトルなし", "\n".join(lines)
            title = lines[0] + " " + lines[1]  # 最初の2行をタイトルに結合
            body = "\n".join(lines[2:])  # 3行目以降を本文に
            return title, body
    except FileNotFoundError:
        print("指定されたファイルが見つかりません！")
        return "エラー", "ファイルが見つかりませんでした！"


# 件名、送信先アドレス、本文を渡してメールオブジェクトを生成する関数
def make_mime_text(mail_to, subject, body):
    # MIMETextで改行を保持
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["To"] = mail_to
    msg["From"] = GMAIL_ACCOUNT
    return msg


# smtp経由でメール送信する関数
def send_gmail(msg):
    try:
        server = smtplib.SMTP_SSL(
            "smtp.gmail.com", 465,
            context=ssl.create_default_context()
        )
        server.set_debuglevel(0)
        server.login(GMAIL_ACCOUNT, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"メール送信中にエラーが発生しました: {e}")


# メインの関数
def send_email(file_path):
    title, body = process_txt_file(file_path)

    # ファイルが見つからなかった場合は処理を終了
    if title == "エラー":
        print("メール送信を中止しました。")
        return

    # メールを作成して送信
    msg = make_mime_text(
        mail_to=GMAIL_SEND_TO,
        subject=title,
        body=body
    )
    send_gmail(msg)
    print("メール送信成功！:", title)


def send_race_pred(race_day, race_id):
    text_path = os.path.join(paths.RACE_PREDICTION_TEXT_PATH, race_day.strftime("%Y%m%d"), f"{race_id}.txt")
    send_email(text_path)


def post_text_error(e):
    """エラー時動作を記載する

    Args:
        e (Exception) : エラー内容
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")


def post_text_data(text_path):
    """テキストを投稿する

    Args:
        text_path(str) : テキストのパス
    """
    # Twitterの認証
    client = tweepy.Client(
        consumer_key=X_API_KEY,
        consumer_secret=X_API_SECRET,
        access_token=X_ACCESS_TOKEN,
        access_token_secret=X_ACCESS_TOKEN_SECRET,
    )
    if os.path.isfile(text_path):
        fp = open(text_path, "r", encoding="utf-8")
        tweet_str = fp.read()
        try:
            # ツイートの実行
            client.create_tweet(text=tweet_str)
        except Exception as e:
            post_text_error(e)
            fp.close()
            raise Exception("post failed")
        fp.close()
    else:
        print("no text file")


def post_text(text):
    """文字列をそのまま投稿する

    post_text_dataはファイル（事前に保存済みの予想・回収率テキスト）経由だが、
    週末まとめ・週末プレビュー等、その場で組み立てた文章を投稿する用途では
    ファイルを経由せずこちらを使う。

    Args:
        text (str): 投稿する本文。
    """
    client = tweepy.Client(
        consumer_key=X_API_KEY,
        consumer_secret=X_API_SECRET,
        access_token=X_ACCESS_TOKEN,
        access_token_secret=X_ACCESS_TOKEN_SECRET,
    )
    try:
        client.create_tweet(text=text)
    except Exception as e:
        post_text_error(e)
        raise Exception("post failed")


def post_text_with_image(text, image_path):
    """画像付きのテキストを投稿する（高配当的中ハイライト等で使う）

    メディアのアップロードはX API v1.1（tweepy.API、OAuth1UserHandler）でのみ
    対応しているため、v2のClient（テキスト投稿）と併用する
    （tweepyの公式な画像付きツイートの実装パターン）。

    Args:
        text (str): 投稿本文。
        image_path (str): アップロードする画像のパス。
    """
    auth = tweepy.OAuth1UserHandler(
        X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET,
    )
    api_v1 = tweepy.API(auth)
    client = tweepy.Client(
        consumer_key=X_API_KEY,
        consumer_secret=X_API_SECRET,
        access_token=X_ACCESS_TOKEN,
        access_token_secret=X_ACCESS_TOKEN_SECRET,
    )
    try:
        media = api_v1.media_upload(image_path)
        client.create_tweet(text=text, media_ids=[media.media_id])
    except Exception as e:
        post_text_error(e)
        raise Exception("post failed")
