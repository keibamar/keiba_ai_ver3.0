import io
import shutil
from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.managers import race_info_dataset_manager, race_schedule_dataset_manager
from src.output import prediction_publisher
from src.output import return_report as rr

SAMPLE_DATE_STR = "20241020"
SAMPLE_RACE_DAY = date(2024, 10, 20)
SAMPLE_PLACE_ID = 4
SAMPLE_RACE_ID = "202404040601"


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    """race_card/配当結果/レポート出力先をtmp_path配下に切り替える。

    race_calendar（data/race_schedule/2024_race_calendar.csv）は
    実データをそのまま参照する（202404040601は2024/10/20の新潟1Rに対応）。
    """
    monkeypatch.setattr(paths, "RACE_CARD_DATA_PATH", str(tmp_path / "race_card"))
    monkeypatch.setattr(paths, "RACE_RETURN_REPORT_TEXT_PATH", str(tmp_path / "texts" / "race_returns"))
    monkeypatch.setattr(race_info_dataset_manager, "RACE_RETURNS_DATA_PATH", str(tmp_path / "race_returns"))

    race_card_dir = tmp_path / "race_card" / SAMPLE_DATE_STR
    race_card_dir.mkdir(parents=True)
    shutil.copy(
        f"data/race_card/{SAMPLE_DATE_STR}/{SAMPLE_RACE_ID}.csv",
        race_card_dir / f"{SAMPLE_RACE_ID}.csv",
    )

    # サンプルレースカードはrank1=馬番5, rank2=馬番9, rank3=馬番7
    returns_dir = tmp_path / "race_returns" / "04_nigata" / "2024"
    returns_dir.mkdir(parents=True)
    returns_df = pd.DataFrame(
        {
            "式別": ["単勝", "複勝", "複勝", "複勝", "三連複"],
            "馬番": ["5", "5", "9", "7", "5-9-7"],
            "配当": ["160", "110", "150", "180", "4220"],
            "人気": ["1", "1", "3", "4", "12"],
        },
        index=[SAMPLE_RACE_ID] * 5,
    )
    returns_df.index.name = ""
    returns_df.to_csv(returns_dir / f"{SAMPLE_RACE_ID}.csv")

    return tmp_path


@pytest.fixture
def race_id_list():
    return race_schedule_dataset_manager.get_daily_id(SAMPLE_PLACE_ID, SAMPLE_RACE_DAY)


class FakeTweepyClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tweets = []
        FakeTweepyClient.instances.append(self)

    def create_tweet(self, text):
        self.tweets.append(text)


def test_get_win_result_hits_rank1_horse(new_roots, race_id_list):
    hit_rate, return_rate, hit_race = rr.get_win_result(SAMPLE_RACE_DAY, race_id_list)
    assert hit_rate == 1
    assert return_rate == 160.0
    assert hit_race == "1 "


def test_get_win_result_uses_top_score_when_rank_is_tied(new_roots, race_id_list):
    # 馬番5(score最大)と馬番8に同じrank=1を付け、tieが発生してもscore最大の
    # 馬番5を本命馬として的中判定すること（rankの値だけで決めると、的中判定が
    # ゆるくなり回収率が実態より高く出てしまう不具合の再発防止）
    race_card_path = new_roots / "race_card" / SAMPLE_DATE_STR / f"{SAMPLE_RACE_ID}.csv"
    df = pd.read_csv(race_card_path, index_col=0)
    df.loc[df["馬番"] == 8, "rank"] = 1
    df.to_csv(race_card_path)

    hit_rate, return_rate, hit_race = rr.get_win_result(SAMPLE_RACE_DAY, race_id_list)
    assert hit_rate == 1
    assert return_rate == 160.0


def test_get_place_result_hits_rank1_horse(new_roots, race_id_list):
    hit_rate, return_rate, hit_race = rr.get_place_result(SAMPLE_RACE_DAY, race_id_list)
    assert hit_rate == 1
    assert return_rate == 110.0
    assert hit_race == "1 "


def test_get_trio_box_result_hits_top3_within_box(new_roots, race_id_list):
    hit_rate, return_rate, hit_race = rr.get_trio_box_result(SAMPLE_RACE_DAY, race_id_list, box_num=5)
    assert hit_rate == 1
    assert return_rate == 422.0
    assert hit_race == "1 "


def test_write_win_hit_text_format(new_roots, race_id_list):
    buf = io.StringIO()
    rr.write_win_hit_text(SAMPLE_RACE_DAY, race_id_list, buf)
    assert buf.getvalue() == "◎単勝回収率:160.0%  (的中レース:1 R)\n"


def test_write_place_hit_text_format(new_roots, race_id_list):
    buf = io.StringIO()
    rr.write_place_hit_text(SAMPLE_RACE_DAY, race_id_list, buf)
    assert buf.getvalue() == "◎複勝回収率:110.0%  (的中レース:1 R)\n"


def test_write_trio_hit_text_format(new_roots, race_id_list):
    buf = io.StringIO()
    rr.write_trio_hit_text(SAMPLE_RACE_DAY, race_id_list, buf)
    assert buf.getvalue() == "三連複(5頭BOX)回収率:422.0%  (的中レース:1 R)\n"


def test_make_return_text_writes_expected_content(new_roots):
    rr.make_return_text(SAMPLE_PLACE_ID, SAMPLE_RACE_DAY)

    out_file = new_roots / "texts" / "race_returns" / SAMPLE_DATE_STR / "04_nigata_pred_score.txt"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_return_text(place_id={SAMPLE_PLACE_ID}, race_day={SAMPLE_RACE_DAY}) ---")
    print(f"  出力先: {out_file}")
    print(content)

    assert content.startswith("2024/10/20 新潟競馬場\n\nMAR競馬予想 回収率\n")
    assert "◎単勝回収率:160.0%  (的中レース:1 R)\n" in content
    assert "◎複勝回収率:110.0%  (的中レース:1 R)\n" in content
    assert "三連複(5頭BOX)回収率:422.0%  (的中レース:1 R)\n" in content
    assert "#MAR競馬予想" in content
    assert "#競馬予想AI" in content
    assert "#競馬 #競馬予想" in content
    assert "#新潟競馬場" in content


def test_post_race_returns_posts_tweet(new_roots, monkeypatch):
    FakeTweepyClient.instances = []
    monkeypatch.setattr(prediction_publisher.tweepy, "Client", FakeTweepyClient)

    rr.make_return_text(SAMPLE_PLACE_ID, SAMPLE_RACE_DAY)
    rr.post_race_returns(SAMPLE_PLACE_ID, SAMPLE_RACE_DAY)

    client = FakeTweepyClient.instances[0] if FakeTweepyClient.instances else None

    print(f"\n--- post_race_returns(place_id={SAMPLE_PLACE_ID}, race_day={SAMPLE_RACE_DAY}) ---")
    print(f"  投稿内容: {client.tweets if client else None}")

    assert len(FakeTweepyClient.instances) == 1
    assert len(client.tweets) == 1
    assert "◎単勝回収率:160.0%" in client.tweets[0]


def test_post_daily_race_returns_posts_for_each_racing_place(new_roots, monkeypatch):
    """開催のあった全place_idについてmake_return_text/post_race_returnsが呼ばれ、
    開催のなかったplace_idはスキップされることを確認する。
    """
    FakeTweepyClient.instances = []
    monkeypatch.setattr(prediction_publisher.tweepy, "Client", FakeTweepyClient)

    calls = []
    monkeypatch.setattr(rr, "make_return_text", lambda place_id, race_day: calls.append(("make_return_text", place_id)))
    monkeypatch.setattr(rr, "post_race_returns", lambda place_id, race_day: calls.append(("post_race_returns", place_id)))

    rr.post_daily_race_returns(SAMPLE_RACE_DAY)

    print(f"\n--- post_daily_race_returns({SAMPLE_RACE_DAY}) ---")
    print(f"  呼び出し: {calls}")

    # 2024/10/20 に開催があったのは 04_nigata(4) / 05_tokyo(5) / 08_kyoto(8) のみ
    racing_place_ids = {4, 5, 8}
    assert {place_id for _, place_id in calls} == racing_place_ids
    assert all(name in ("make_return_text", "post_race_returns") for name, _ in calls)
