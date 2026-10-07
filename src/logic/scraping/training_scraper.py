"""netkeiba プレミアム会員向け調教タイムスクレイパー

ログインが必要なページ:
    https://db.netkeiba.com/horse/training.html?id={horse_id}

使い方:
    from src.logic.scraping.training_scraper import login_netkeiba, scrape_horse_training
    session = login_netkeiba(login_id, password)      # 認証情報は引数で渡す。コードに書かない
    df = scrape_horse_training('2019102632', session)

HTML構造メモ:
    各テーブルの <caption> にレース情報（日付・開催・レース名・結果）
    調教タイムは <ul class="TrainingTimeDataList"> の各 <li> に格納
    相手馬情報は <p class="TrainingHeisou"> (省略可)
"""

import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.logic.scraping.common import scraping_header


def login_netkeiba(login_id: str, password: str) -> requests.Session:
    """netkeibaにログインして認証済みSessionを返す。"""
    session = requests.Session()
    resp = session.post(
        "https://regist.netkeiba.com/",
        data={
            "pid": "login",
            "action": "auth",
            "rtn_url": "",
            "login_id": login_id,
            "pswd": password,
        },
        headers=scraping_header,
        timeout=15,
    )
    resp.raise_for_status()
    return session


def scrape_horse_training(horse_id: str, session: requests.Session) -> pd.DataFrame:
    """horse_idの全調教タイムを全ページ取得してDataFrameで返す。

    ページネーションは POST id/mode/page で制御される。
    pagerの「次」リンクがなくなるまで全ページを取得する。

    Returns:
        DataFrame columns:
            horse_id, race_id, race_date, race_venue, race_name, race_result,
            trainer_comment, training_date, training_time_of_day, course,
            track_condition, rider, direction, times_list, num_poles,
            position, leg_type, evaluation, grade, partner_info
    """
    records = []
    page = 1

    while True:
        if page == 1:
            url = f"https://db.netkeiba.com/horse/training.html?id={horse_id}"
            resp = session.get(url, headers=scraping_header, timeout=15)
        else:
            resp = session.post(
                "https://db.netkeiba.com/horse/training.html",
                data={"id": horse_id, "mode": "", "page": str(page)},
                headers=scraping_header,
                timeout=15,
            )
        resp.raise_for_status()

        enc = resp.apparent_encoding or "EUC-JP"
        soup = BeautifulSoup(resp.content.decode(enc, "ignore"), "html.parser")

        tables = soup.find_all("table")
        if not tables:
            break

        for table in tables:
            race_id, race_date, race_venue, race_name, race_result = _parse_caption(table)
            trainer_comment = _extract_trainer_comment(table)

            rows = table.find_all("tr")
            for row in rows[2:]:  # row[0]=headers, row[1]=trainer comment
                tds = row.find_all("td")
                if len(tds) < 9:
                    continue

                training_date, training_time_of_day = _parse_datetime(
                    tds[0].get_text(strip=True)
                )
                course = tds[1].get_text(strip=True)
                track_condition = tds[2].get_text(strip=True)
                rider = tds[3].get_text(strip=True)

                direction, times_list = _parse_time_cell(tds[4])
                partner_info = _extract_partner_info(tds[4])

                position = tds[5].get_text(strip=True)
                leg_type = tds[6].get_text(strip=True)
                evaluation = tds[7].get_text(strip=True)
                grade = tds[8].get_text(strip=True) if len(tds) > 8 else ""

                records.append(
                    {
                        "horse_id": horse_id,
                        "race_id": race_id,
                        "race_date": race_date,
                        "race_venue": race_venue,
                        "race_name": race_name,
                        "race_result": race_result,
                        "trainer_comment": trainer_comment,
                        "training_date": training_date,
                        "training_time_of_day": training_time_of_day,
                        "course": course,
                        "track_condition": track_condition,
                        "rider": rider,
                        "direction": direction,
                        "times_list": times_list,
                        "num_poles": len(times_list),
                        "position": position,
                        "leg_type": leg_type,
                        "evaluation": evaluation,
                        "grade": grade,
                        "partner_info": partner_info,
                    }
                )

        # 「次」リンクがなければ最終ページ
        has_next = any(
            "次" in a.get_text() and "paging" in a.get("href", "")
            for a in soup.find_all("a", href=True)
        )
        if not has_next:
            break
        page += 1
        time.sleep(1.0)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["training_date"] = pd.to_datetime(df["training_date"], errors="coerce")
    return df


def _parse_caption(table) -> tuple[str, str, str, str, str]:
    """<caption>からレース情報を抽出する。

    例: "2026/10/04  東京11R  毎日王冠(GII)  結果 ： 1着"
    """
    cap = table.find("caption")
    if not cap:
        return "", "", "", "", ""

    race_name = ""
    race_id = ""
    a = cap.find("a", href=True)
    if a:
        race_name = a.get_text(strip=True)
        m = re.search(r"race/(\d+)/", a["href"])
        if m:
            race_id = m.group(1)

    cap_text = cap.get_text(" ", strip=True)

    race_date = ""
    m = re.search(r"(\d{4}/\d{2}/\d{2})", cap_text)
    if m:
        race_date = m.group(1).replace("/", "-")

    race_venue = ""
    m = re.search(r"\d{4}/\d{2}/\d{2}\s+(\S+\d+[Rr])", cap_text)
    if m:
        race_venue = m.group(1)

    race_result = ""
    m = re.search(r"結果\s*[：:]\s*(\S+)", cap_text)
    if m:
        race_result = m.group(1)

    return race_id, race_date, race_venue, race_name, race_result


def _extract_trainer_comment(table) -> str:
    """テーブルrow[1]（短評行）からtrainerコメントを取得する。"""
    rows = table.find_all("tr")
    if len(rows) < 2:
        return ""
    text = rows[1].get_text(strip=True)
    return re.sub(r"^\[短評\]", "", text).strip()


def _parse_datetime(date_raw: str) -> tuple[str, str]:
    """'2026/10/01 05:30:00(木)' → ('2026-10-01', '05:30:00')"""
    m = re.match(r"(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})", date_raw)
    if m:
        return m.group(1).replace("/", "-"), m.group(2)
    m2 = re.match(r"(\d{4}/\d{2}/\d{2})", date_raw)
    if m2:
        return m2.group(1).replace("/", "-"), ""
    return date_raw, ""


def _parse_time_cell(td) -> tuple[str, list[float]]:
    """<td class="TrainingTimeData"> から (direction, times_list) を返す。

    <ul class="TrainingTimeDataList">
      <li>-</li>              ← 方向 ("-"=内, "+"=外 など) — 省略されることもある
      <li class="TokeiColor02">53.1</li>
      ...
      <li class="TokeiColor01">12.4</li>
    </ul>
    """
    ul = td.find("ul", class_="TrainingTimeDataList")
    if not ul:
        return "", []

    direction = ""
    times_list = []
    for li in ul.find_all("li"):
        txt = li.get_text(strip=True)
        if re.match(r"^\d+\.\d$", txt):
            times_list.append(float(txt))
        elif txt in ("-", "+", "内", "外"):
            direction = txt

    return direction, times_list


def _extract_partner_info(td) -> str:
    """<p class="TrainingHeisou"> から相手馬情報を取得する。"""
    p = td.find("p", class_="TrainingHeisou")
    return p.get_text(strip=True) if p else ""


def scrape_multiple_horses(
    horse_ids: list[str],
    session: requests.Session,
    sleep_sec: float = 1.5,
) -> pd.DataFrame:
    """複数horse_idの調教データをまとめて取得する。"""
    dfs = []
    for i, horse_id in enumerate(horse_ids):
        print(f"[{i+1}/{len(horse_ids)}] {horse_id}")
        try:
            df = scrape_horse_training(horse_id, session)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"  ERROR: {e}")
        if i < len(horse_ids) - 1:
            time.sleep(sleep_sec)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
