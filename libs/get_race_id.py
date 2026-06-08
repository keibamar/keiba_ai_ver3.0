import sys

from datetime import date, timedelta
import pandas as pd

# pycache を生成しない
sys.dont_write_bytecode = True

# race_calendarのパス
CALENDAR_PATH = "C:/keiba_ai/keiba_ai_ver2.0/texts/race_calendar/"

def get_race_id_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def get_year_id_all(place_id, year = date.today().year):
    """ 開催コースと年を指定して、1年間のrace_idを取得 
    Args:
        place_id (int) : 開催コースid
        year(int) : 年（初期値：今年）
    Returns:
        list: 指定した、開催コース、年のstr型のrace_idを返す    
    """
    race_id_list = []
    for times in range(1, 7, 1): # 開催数(6)
        for race_day in range(1, 13, 1): #開催日(12)
            for race in range(1, 13, 1): #レース数(12)
                race_id = str(year) + str(place_id).zfill(2) + str(times).zfill(2) + str(race_day).zfill(2) + str(race).zfill(2)
                race_id_list.append(race_id)
    return race_id_list

def get_year_id_calendar(place_id, year = date.today().year):
    """ 開催コースと年を指定して、1年間のrace_idをレースカレンダーから取得 
        Args:
            place_id (int) : 開催コースid
            year(int) : 年（初期値：今年）
        Returns:
            list: 指定した、開催コース、年のstr型のrace_idを返す    
    """
    race_id_list = []
    try:
        race_calendar = pd.read_csv(CALENDAR_PATH + str(year) + "_race_calendar.csv", dtype = str)
        race_data = race_calendar[race_calendar['course'] == str(place_id)].reset_index(drop = True)
        # race_idリストに変換
        ### id = [開催年][競馬場][開催][開催日][レース]
        if not race_data.empty:
            for i in range(len(race_data.index)):
                for race in range(1, 13):  
                    id = str(year) + str(race_data.at[0,"course"]).zfill(2) + str(race_data.at[0,"times"]).zfill(2) + str(race_data.at[0,"days"]).zfill(2) + str(race).zfill(2)
                    race_id_list.append(id)
        return race_id_list
    # エラー時　エラー内容を出力
    except Exception as e:
        get_race_id_error(e)
        return race_id_list

def get_past_year_id(place_id = 0, race_day = date.today()):
    """ 指定した日にちまでの、その年のrace_idを取得 
        Args:
            place_id (int) : 開催コースid (初期値0=全コース)
            race_day(Date) : 日（初期値：今日）
        Returns:
            list: 指定した日にちまでの、その年のstr型のrace_idを返す    
    """
    race_id_list = []
    # レースカレンダーを開く
    try:
        race_calendar = pd.read_csv(CALENDAR_PATH + str(race_day.year) + "_race_calendar.csv", dtype = str)
    
        today_races = []
        # 指定日までのレースIDを取得
        for m in range(1, race_day.month + 1):
            if m < race_day.month:
                last_day = 31
            else :
                last_day = race_day.day
            for d in range(1, last_day + 1):
                temp = race_calendar[race_calendar['month'] == str(m)]
                temp = temp[temp['day'] == str(str(d))]
                # 開催コース未指定の場合、全コース取得
                if place_id == 0:
                    for ids in range(1, 11):
                        today_race = temp[temp['course'] == str(ids)]
                        today_races.append(today_race.reset_index())
                else:
                    today_race = temp[temp['course'] == str(place_id)]
                    today_races.append(today_race.reset_index())

        # race_idリストに変換
        ### id = [開催年][競馬場][開催][開催日][レース]
        for race_data in today_races:
            if not race_data.empty:
                for race in range(1, 13):  
                    id = str(race_day.year) + str(race_data.at[0,"course"]).zfill(2) + str(race_data.at[0,"times"]).zfill(2) + str(race_data.at[0,"days"]).zfill(2) + str(race).zfill(2)
                    race_id_list.append(id)
        return race_id_list
    # エラー時　エラー内容を出力
    except Exception as e:
        get_race_id_error(e)
        return race_id_list

def get_past_weekly_id(place_id = 0, base_day = date.today()):
    """ 指定した日にちから、1週間前のrace_idを取得 
        Args:
            place_id (int) : 開催コースid (初期値0=全コース)
            base_day(Date) : 日（初期値：今日）
        Returns:
            list: 指定した日にちから、1週間前までのstr型のrace_idを返す    
    """
    race_id_list = []
    # レースカレンダーを開く
    try:
        # 1週間分の開催日を取得
        for i in range(7):
            race_day = base_day - timedelta(days = (7 - i))
            # 今日のレースIDを取得
            race_id_list.append(get_daily_id(place_id, race_day))
        # リストの一次元化
        race_id_list = sum(race_id_list, [])
        return race_id_list
    # エラー時　エラー内容を出力
    except Exception as e:
        get_race_id_error(e)
        return race_id_list
    
def get_next_weekly_id(place_id = 0, race_day = date.today()):
    """ 指定した日にちから、次の1週間のrace_idを取得 
        Args:
            place_id (int) : 開催コースid (初期値0=全コース)
            race_day(Date) : 日（初期値：今日）
        Returns:
            list: 指定した日にちから、次の1週間のstr型のrace_idを返す    
    """
    race_id_list = []
    # レースカレンダーを開く
    try:
        # 1週間分の開催日を取得
        for i in range(7):
            race_day = race_day + timedelta(days = (i))
            # 今日のレースIDを取得
            race_id_list.append(get_daily_id(place_id, race_day))
        # リストの一次元化
        race_id_list = sum(race_id_list, [])
        return race_id_list
    # エラー時　エラー内容を出力
    except Exception as e:
        get_race_id_error(e)
        return race_id_list

def get_daily_id(place_id = 0, race_day = date.today()):
    """ 指定した日にちのrace_idを取得 
        Args:
            place_id (int) : 開催コースid (初期値0=全コース)
            race_day(Date) : 日（初期値：今日）
        Returns:
            list: 指定した日にちのstr型のrace_idを返す    
    """
    race_id_list = []
    # レースカレンダーを開く
    try:
        race_calendar = pd.read_csv(CALENDAR_PATH + str(race_day.year) + "_race_calendar.csv", dtype = str)
    
        today_races = []
        # 今日のレースIDを取得
        temp = race_calendar[race_calendar['month'] == str(race_day.month)]
        temp = temp[temp['day'] == str(race_day.day)]

        # 開催コース未指定の場合、全コース取得
        if place_id == 0:
            for ids in range(1, 11):
                today_race = temp[temp['course'] == str(ids)]
                today_races.append(today_race.reset_index())
        else:
            today_race = temp[temp['course'] == str(place_id)]
            today_races.append(today_race.reset_index())

        # race_idリストに変換
        ### id = [開催年][競馬場][開催][開催日][レース]
        for race_data in today_races:
            if not race_data.empty:
                for race in range(1, 13):  
                    id = str(race_day.year) + str(race_data.at[0,"course"]).zfill(2) + str(race_data.at[0,"times"]).zfill(2) + str(race_data.at[0,"days"]).zfill(2) + str(race).zfill(2)
                    race_id_list.append(id)
        return race_id_list
    # エラー時　エラー内容を出力
    except Exception as e:
        get_race_id_error(e)
        return race_id_list

def get_place_id_list_from_race_id_list(race_id_list):
    """race_id_listからplace_id_listを抽出する
        Args:
            race_id_list(list) : race_idのリスト
        Returns:
            race_id_list(list) : race_idのリスト
    """
    place_id_list = []
    for i in range(len(race_id_list)):
        place_id = int(str(race_id_list[i])[4] + str(race_id_list[i])[5])
        place_id_list.append(place_id)
    place_id_list = list(set(place_id_list))
    place_id_list.sort()

    return place_id_list


def get_past_weekly_place_id(race_day = date.today()):
    """指定日から1週間前までの開催場のplace_idを取得
        Args:
            race_day(Date) : 日（初期値：今日）
        Returns :
            place_id_list(list) : 開催場のplace_id
    """
    race_id_list = get_past_weekly_id(base_day = race_day)
    place_id_list = get_place_id_list_from_race_id_list(race_id_list)
    return place_id_list

def get_daily_place_id(race_day = date.today()):
    """指定日の開催場のplace_idを取得
        Args:
            race_day(Date) : 日（初期値：今日）
        Returns :
            place_id_list(list) : 開催場のplace_id
    """
    race_id_list = get_daily_id(race_day = race_day)
    place_id_list = get_place_id_list_from_race_id_list(race_id_list)
    return place_id_list


if __name__ == "__main__":
    now = date.today() -timedelta(1)
    print(get_past_weekly_place_id())
        

