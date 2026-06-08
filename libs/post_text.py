import os
import sys

import tweepy
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True

API_KEY = "Hp0u29yZGFzIWvlKgyUGkABun"
API_SECRET = "2SzoytDpmzaHDfRXm4Pp4rJXpbYfi28obgGd9K3z43U7Y1GPGv"
ACCESS_KEY = "1764156736586514432-CDRClN7pEsZcfVFmpDzOIJuMQ62P1K"
ACCESS_SECRET = "4ICSIwirvOPDxCEafpx365NGisCxiVWsITOj7jCWbwllA"

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