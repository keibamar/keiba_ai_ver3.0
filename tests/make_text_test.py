import re
import sys

import datetime
from datetime import date, timedelta
from time import sleep
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import get_race_id
import name_header
import post_text
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\RacePrediction")
import make_text
import race_card
import day_race_prediction
import calc_returns

def make_race_prediction_text_test():
    return

def make_race_return_text_test():
    return

if __name__ == "__main__":
    print("[0]予想のみ/[1]予想+配当結果")
    test_type = input(":")

    print("日付を指定してください。[yyyymmdd]")
    print("複数の日付を指定する場合は「,」で区切って入力してください。")
    str_race_day = input(":")

    print("に開催されたレースはありません。")
    print("再度日付を指定しますか。[y/n]")
    reset_flag_day = input(":")

    print("の開催場は以下になります。どの開催場の予想を出力しますか。")
    course_type = input(":")

    print("予想するレースを指定してください。")
    race_type = input(":")

    print("指定されたフォーマットで入力してください。")