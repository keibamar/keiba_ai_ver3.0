import requests
from bs4 import BeautifulSoup

def get_jra_schedule(url):
    # ウェブページの内容を取得
    response = requests.get(url)
    response.raise_for_status()  # ステータスコードが200以外の場合に例外を発生させる
    
    # BeautifulSoupを使ってHTMLをパース
    soup = BeautifulSoup(response.text, 'html.parser')

    # テーブルを探してみる
    table = soup.find('table')
    if not table:
        print("テーブルが見つかりませんでした")
        return []

    # 行を取得
    rows = table.find_all('tr')

    schedule_data = []
    for row in rows[1:]:  # ヘッダー行をスキップ
        cols = row.find_all('td')
        if len(cols) > 1:
            day = cols[0].text.strip()
            course = cols[1].text.strip()
            event = cols[2].text.strip()
            day_num = cols[3].text.strip()
            schedule_data.append([day, course, event, day_num])
    
    return schedule_data

# 使用例
url = 'https://www.jra.go.jp/keiba/calendar2025/oct.html'
schedule = get_jra_schedule(url)
for entry in schedule:
    print(entry)
