import shutil
from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.config.constants import NAME_LIST
from src.config.lists import SYMBOL_LIST
from src.managers import race_card_dataset_manager
from src.output import prediction_publisher as pub

SAMPLE_DATE_STR = "20241020"
SAMPLE_RACE_DAY = date(2024, 10, 20)
SAMPLE_PLACE_ID = 4
SAMPLE_RACE_ID = "202404040601"


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    """race_card/予想テキストの出力先をtmp_path配下に切り替える。

    race_time_id_list（texts/race_calendar/race_time_id_list/20241020.csv）は
    実データをそのまま参照する。
    """
    monkeypatch.setattr(paths, "RACE_CARD_DATA_PATH", str(tmp_path / "race_card"))
    monkeypatch.setattr(paths, "RACE_PREDICTION_TEXT_PATH", str(tmp_path / "texts" / "race_prediction"))

    race_card_dir = tmp_path / "race_card" / SAMPLE_DATE_STR
    race_card_dir.mkdir(parents=True)
    shutil.copy(
        f"data/RaceCards/{SAMPLE_DATE_STR}/{SAMPLE_RACE_ID}.csv",
        race_card_dir / f"{SAMPLE_RACE_ID}.csv",
    )
    return tmp_path


class FakeSMTP:
    instances = []

    def __init__(self, host, port, context=None):
        self.host = host
        self.port = port
        self.context = context
        self.calls = []
        FakeSMTP.instances.append(self)

    def set_debuglevel(self, level):
        self.calls.append(("set_debuglevel", level))

    def login(self, account, password):
        self.calls.append(("login", account, password))

    def send_message(self, msg):
        self.calls.append(("send_message", msg))

    def quit(self):
        self.calls.append(("quit",))


class FakeTweepyClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tweets = []
        FakeTweepyClient.instances.append(self)

    def create_tweet(self, text):
        self.tweets.append(text)


def test_extract_top5_pred_returns_top5_in_rank_order():
    df = pd.DataFrame({
        "馬番": [1, 2, 3, 4, 5, 6],
        "馬名": ["A", "B", "C", "D", "E", "F"],
        "rank": [3, 1, 5, 2, 4, 6],
    })
    result = pub.extract_top5_pred(df)
    assert result == [[2, "B"], [4, "D"], [1, "A"], [5, "E"], [3, "C"]]


def test_make_race_text_writes_expected_content(new_roots):
    pub.make_race_text(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)

    out_file = new_roots / "texts" / "race_prediction" / SAMPLE_DATE_STR / f"{SAMPLE_RACE_ID}.txt"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    lines = content.splitlines()

    print(f"\n--- make_race_text({SAMPLE_RACE_DAY}, race_id={SAMPLE_RACE_ID}) ---")
    print(f"  出力先: {out_file}")
    print(content)

    # 日付行
    assert lines[0] == "2024/10/20"

    # 開催情報行（場名+R番+race_name+発走時刻）
    time_id_df = race_card_dataset_manager.get_race_time_id_list_df(SAMPLE_RACE_DAY)
    row = time_id_df[time_id_df["race_id"] == SAMPLE_RACE_ID].iloc[0]
    expected_start_time = row["race_time"][:2] + ":" + row["race_time"][2:]
    expected_header = f"{NAME_LIST[SAMPLE_PLACE_ID - 1]}1R {row['race_name']} {expected_start_time}"
    assert expected_header in content

    # 予想印 + 上位5頭（馬番・馬名）
    race_card_df = race_card_dataset_manager.get_race_cards(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)
    pred_list = pub.extract_top5_pred(race_card_df)
    assert pred_list  # サンプルデータにはrank1〜5が存在する
    for i, (num, name) in enumerate(pred_list):
        assert f" {SYMBOL_LIST[i]} {num} {name}" in content

    # ハッシュタグ
    assert "#MAR競馬予想" in content
    assert "#競馬予想AI" in content
    assert "#競馬 #競馬予想" in content
    assert f"#{NAME_LIST[SAMPLE_PLACE_ID - 1]}競馬場" in content


def test_make_race_text_skips_when_no_rank_column(new_roots):
    race_card_df = race_card_dataset_manager.get_race_cards(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)
    race_card_df = race_card_df.drop(columns=["rank"])
    race_card_dataset_manager.save_race_cards(race_card_df, SAMPLE_RACE_DAY, SAMPLE_RACE_ID)

    pub.make_race_text(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)

    out_file = new_roots / "texts" / "race_prediction" / SAMPLE_DATE_STR / f"{SAMPLE_RACE_ID}.txt"
    assert not out_file.exists()


def test_send_email_sends_via_smtp(tmp_path, monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(pub.smtplib, "SMTP_SSL", FakeSMTP)

    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("2024/10/20\n新潟1R レース 09:50\n\n本文1行目\n本文2行目\n", encoding="utf-8")

    pub.send_email(str(txt_path))

    assert len(FakeSMTP.instances) == 1
    smtp = FakeSMTP.instances[0]
    call_names = [call[0] for call in smtp.calls]

    sent_msg = smtp.calls[2][1]

    print(f"\n--- send_email({txt_path}) ---")
    print(f"  SMTP呼び出し順: {call_names}")
    print(f"  件名: {sent_msg['Subject']}")
    print(f"  本文: {sent_msg.get_payload(decode=True).decode('utf-8')!r}")

    assert call_names == ["set_debuglevel", "login", "send_message", "quit"]
    assert sent_msg["Subject"] == "2024/10/20 新潟1R レース 09:50"
    assert sent_msg.get_payload(decode=True).decode("utf-8") == "本文1行目\n本文2行目"


def test_send_email_missing_file_does_not_send(tmp_path, monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(pub.smtplib, "SMTP_SSL", FakeSMTP)

    pub.send_email(str(tmp_path / "missing.txt"))

    assert FakeSMTP.instances == []


def test_post_text_data_creates_tweet(tmp_path, monkeypatch):
    FakeTweepyClient.instances = []
    monkeypatch.setattr(pub.tweepy, "Client", FakeTweepyClient)

    txt_path = tmp_path / "tweet.txt"
    txt_path.write_text("テスト投稿\n#MAR競馬予想\n", encoding="utf-8")

    pub.post_text_data(str(txt_path))

    client = FakeTweepyClient.instances[0] if FakeTweepyClient.instances else None

    print(f"\n--- post_text_data({txt_path}) ---")
    print(f"  投稿内容: {client.tweets if client else None}")

    assert len(FakeTweepyClient.instances) == 1
    assert client.tweets == ["テスト投稿\n#MAR競馬予想\n"]


def test_post_text_data_no_file_does_not_raise(tmp_path, monkeypatch):
    FakeTweepyClient.instances = []
    monkeypatch.setattr(pub.tweepy, "Client", FakeTweepyClient)

    pub.post_text_data(str(tmp_path / "missing.txt"))

    assert len(FakeTweepyClient.instances) == 1
    assert FakeTweepyClient.instances[0].tweets == []
