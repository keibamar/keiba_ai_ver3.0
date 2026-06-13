import os
import sys

import tweepy
from dotenv import find_dotenv, load_dotenv
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True

load_dotenv(find_dotenv())

# X(Twitter) APIキー（.envのX_API_KEY等で設定）
API_KEY = os.environ["X_API_KEY"]
API_SECRET = os.environ["X_API_SECRET"]
ACCESS_KEY = os.environ["X_ACCESS_TOKEN"]
ACCESS_SECRET = os.environ["X_ACCESS_TOKEN_SECRET"]

def post_text_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def post_text_data(text_path):
    """ テキストを投稿する 
        Args:
            text_path(str) : テキストのパス 
    """
    #Twitterの認証
    client = tweepy.Client(consumer_key=API_KEY, consumer_secret=API_SECRET, access_token=ACCESS_KEY, access_token_secret=ACCESS_SECRET)
    if os.path.isfile(text_path):
        fp = open(text_path, "r", encoding="utf-8")
        tweet_str = fp.read()
        try:
            # ツイートの実行
            client.create_tweet(text = tweet_str)
        except Exception as e:
            post_text_error(e)
            # print("post failed")
            fp.close()
            raise Exception("post failed")
        fp.close()
    else :
        print("no text file")