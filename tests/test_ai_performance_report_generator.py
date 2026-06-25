"""src/logic/html_generator/ai_performance_report_generator.py のテスト（オフライン）。

ai_performance_dataset_manager.get_ai_performance_dataset をmonkeypatchして
既知のデータセットを与え、ページ生成（出力先・内容・内訳テーブル）を検証する。

既知データセット（SAMPLE_DF、東京=place_id 5、新潟=place_id 9）:
  A: 2025年 東京1回 芝1400m 良    未勝利   win=1/200 place=1/150 trio=0/0
  B: 2026年 東京2回 芝1400m 稍重  1勝クラス win=0/0   place=0/0   trio=1/300
  C: 2026年 東京3回 ダート1600m 良 未勝利   win=1/100 place=1/120 trio=0/0
  D: 2024年 新潟1回 芝1400m 良    未勝利   win=0/0   place=0/0   trio=0/0
今日の日付は2026年（currentDateの環境前提）のため、「今年の成績」は2026年で計算される。
"""

from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.logic.html_generator import ai_performance_report_generator as r
from src.logic.html_generator.rate_gauge_html import hit_rate_gauge_html, return_rate_big_html, return_rate_gauge_html

SAMPLE_DF = pd.DataFrame(
    {
        "race_day": ["2025-05-01", "2026-04-01", "2026-04-08", "2024-03-01"],
        "year": ["2025", "2026", "2026", "2024"],
        "place_id": ["5", "5", "5", "9"],
        "times": ["1", "2", "3", "1"],
        "race_type": ["芝", "芝", "ダート", "芝"],
        "course_len": ["1400", "1400", "1600", "1400"],
        "ground_state": ["良", "稍重", "良", "良"],
        "class": ["未勝利", "1勝クラス", "未勝利", "未勝利"],
        "win_hit": ["1", "0", "1", "0"],
        "win_return": ["200.0", "0.0", "100.0", "0.0"],
        "place_hit": ["1", "0", "1", "0"],
        "place_return": ["150.0", "0.0", "120.0", "0.0"],
        "trio_box_hit": ["0", "1", "0", "0"],
        "trio_box_return": ["0.0", "300.0", "0.0", "0.0"],
    },
    index=["A", "B", "C", "D"],
)


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PUBLIC_HTML_PERFORMANCE_PATH", str(tmp_path / "public_html" / "performance"))
    return tmp_path


@pytest.fixture
def fake_dataset(monkeypatch):
    monkeypatch.setattr(r.m, "get_ai_performance_dataset", lambda: SAMPLE_DF)


@pytest.fixture
def empty_dataset(monkeypatch):
    monkeypatch.setattr(r.m, "get_ai_performance_dataset", lambda: pd.DataFrame())


def test_make_ai_performance_index_page_generates_html(new_roots, fake_dataset):
    r.make_ai_performance_index_page()

    out_file = new_roots / "public_html" / "performance" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_ai_performance_index_page() ---")
    print(html_content)

    assert "<h1>AI予想成績</h1>" in html_content
    assert "<h3>トータル成績</h3>" in html_content
    assert "<h3>2026年の成績</h3>" in html_content
    # トータル成績（4レース全体）: win hit=2/4=50.0%, return=(200+0+100+0)/4=75.0%
    # 回収率を主役（大きな数字+大きめのゲージ）、的中率を脇役（小さなゲージ）にしたカード表示
    assert '<div class="bet-stat-label">単勝</div>' in html_content
    assert return_rate_big_html(75.0) in html_content
    assert '<div class="bet-stat-primary-gauge">' in html_content
    assert hit_rate_gauge_html(50.0) in html_content
    assert "対象 4件" in html_content
    # 新しい年から順にリンクされる
    assert html_content.index("annual/2026.html") < html_content.index("annual/2025.html") < html_content.index("annual/2024.html")
    assert '<a href="annual/2026.html">2026年</a>' in html_content
    assert '<a href="course/05_tokyo/index.html">東京</a>' in html_content
    assert '<nav class="site-nav">' in html_content
    assert '<script src="../assets/js/sortable-table.js"></script>' in html_content
    # トップ階層はブレッドクラムのみ（サイドバーは追加しない）
    assert '<p class="breadcrumb">' in html_content
    assert '<span class="breadcrumb-current">AI成績</span>' in html_content
    assert '<aside class="page-sidebar">' not in html_content
    # 右側タブには、AI成績の直下に全競馬場が選択肢として表示される
    assert '<a href="../performance/course/05_tokyo/index.html">東京</a>' in html_content
    assert '<a href="../performance/course/06_nakayama/index.html">中山</a>' in html_content
    # 「今年の成績の推移」（年間成績より上）と「年間成績」内の年次トレンドの両方に、
    # 単勝・複勝の的中率・回収率グラフ（左軸=的中率、右軸=回収率）が追加されている
    # （週別トレンド2つ + 年次トレンド2つ = 合計4つのグラフ）
    assert "<h2>今年の成績の推移</h2>" in html_content
    assert html_content.index("今年の成績の推移") < html_content.index("年間成績")
    assert '<div class="trend-section">' in html_content
    assert '<svg class="trend-chart"' in html_content
    assert html_content.count('<svg class="trend-chart"') == 4
    assert "単勝 的中率・回収率の推移" in html_content
    assert "複勝 的中率・回収率の推移" in html_content


def test_make_ai_performance_index_page_handles_empty_dataset(new_roots, empty_dataset):
    r.make_ai_performance_index_page()

    out_file = new_roots / "public_html" / "performance" / "index.html"
    html_content = out_file.read_text(encoding="utf-8")
    assert "予想データがまだありません。" in html_content
    assert '<svg class="trend-chart"' not in html_content


def test_make_annual_performance_page_generates_html(new_roots, fake_dataset):
    r.make_annual_performance_page(2026)

    out_file = new_roots / "public_html" / "performance" / "annual" / "2026.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_annual_performance_page(2026) ---")
    print(html_content)

    assert "<h1>2026年 AI予想成績</h1>" in html_content
    # 2026年はB・Cの2レース: win hit=1/2=50.0%, return=(0+100)/2=50.0
    assert return_rate_big_html(50.0) in html_content
    assert hit_rate_gauge_html(50.0) in html_content
    # ブレッドクラムと、他の年度へのサイドバーが追加されている
    assert '<a href="../../performance/index.html">AI成績</a>' in html_content
    assert '<span class="breadcrumb-current">2026年</span>' in html_content
    assert '<aside class="page-sidebar">' in html_content
    assert '<p class="page-sidebar-up"><a href="../index.html">&uarr; AI成績トップ</a></p>' in html_content
    assert "<h3>年度</h3>" in html_content
    assert '<li><a href="2025.html">2025年</a></li>' in html_content
    assert '<li><span class="page-sidebar-current">2026年</span></li>' in html_content
    # 開催週ごとの傾向・推移（B=2026-04-01週、C=2026-04-08週は別の週開始日になる）
    assert "<h2>開催週別の傾向・推移</h2>" in html_content
    assert html_content.count('<svg class="trend-chart"') == 2
    # グラフの横軸は年を省いた短縮表記（週開始日テーブルは元のまま完全な日付を保持する）
    assert ">3/28</text>" in html_content
    assert ">4/4</text>" in html_content
    assert "<h3>開催週別成績</h3>" in html_content
    assert "<td>2026-03-28</td>" in html_content
    assert "<td>2026-04-04</td>" in html_content


def test_make_meeting_performance_page_generates_html(new_roots, fake_dataset):
    r.make_meeting_performance_page(2025, place_id=5, times=1)

    out_file = new_roots / "public_html" / "performance" / "meeting" / "2025" / "05_tokyo-1th.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")
    assert "<h1>2025年 東京1回 AI予想成績</h1>" in html_content
    # 2025年東京1回はAのみ: win hit=1/1=100.0%, return=200.0
    assert return_rate_big_html(200.0) in html_content
    assert hit_rate_gauge_html(100.0) in html_content
    # 開催別ページはブレッドクラムのみ（サイドバーは追加しない）
    assert '<p class="breadcrumb">' in html_content
    assert '<span class="breadcrumb-current">2025年 東京1回</span>' in html_content
    assert '<aside class="page-sidebar">' not in html_content
    assert "<h2>レース詳細</h2>" in html_content


def test_make_meeting_performance_page_shows_real_race_details_newest_day_first(new_roots):
    # 東京2026年第3回（6/6・6/7・6/13・6/14・6/20・6/21開催）。実データに基づく
    # レース詳細（ai_performance_calculator.get_meeting_race_details）が表示される。
    # （fake_datasetは使わない。レース詳細は実データセット側のrace_day・race_idを
    # そのまま使うため、totalも実データになる）
    r.make_meeting_performance_page(2026, place_id=5, times=3)

    out_file = new_roots / "public_html" / "performance" / "meeting" / "2026" / "05_tokyo-3th.html"
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_meeting_performance_page(2026, 5, 3) レース詳細 ---")
    print(html_content)

    # トータル成績の対象レース数（68件）と、レース詳細に並ぶレース数が一致する
    # （以前は確定結果データが欠けている開催日のレースが詳細から漏れ、68件のうち
    # 一部しか表示されない不具合があった）
    assert "<div class=\"bet-stat-n\">対象 68件</div>" in html_content
    # 開催日は新しい順（6/21が最初に出てくる）。確定結果データが欠けている
    # 6/6・6/7も、結果が無い項目は「-」表示で詳細から漏れずに表示される
    assert html_content.index("2026/06/21") < html_content.index("2026/06/13")
    assert "2026/06/06" in html_content
    assert "2026/06/07" in html_content
    # 東京11R(06-13 ジューンS)はAI本命馬カネラフィーナが1着で単勝・複勝とも的中する
    assert "ジューンS" in html_content
    assert html_content.count("カネラフィーナ") == 2  # 勝ち馬・AI本命馬の両方の列に表示される
    assert '<span class="hit-badge win">的中</span> 510円' in html_content
    assert '<span class="hit-badge win">的中</span> 210円' in html_content
    # 人気(3番人気)・着順(1着)も、レース結果ページと同じ配色（RANK_COLORS）で強調する
    assert '<td style="background-color:#FFA07A;">3</td>' in html_content
    assert '<td style="background-color:#FFD700;">1</td>' in html_content


def test_meeting_race_detail_table_html_shows_dash_when_pick_scratched():
    days = [{
        "race_day": date(2026, 6, 13),
        "races": [{
            "race_id": "202605030311", "race_name": "テストS", "race_type": "芝", "course_len": "1800",
            "ground_state": "良", "class": "オープン",
            "winner_name": "ホースB", "pick_name": "ホースA", "pick_pop": "1", "pick_finish": "除外",
            "pick_scratched": True, "win_hit": False, "win_payout": None,
            "place_hit": False, "place_payout": None,
        }],
    }]

    html_content = r._meeting_race_detail_table_html(days)

    print(f"\n--- _meeting_race_detail_table_html（本命馬除外）---\n{html_content}")

    # 着順列には「除外」をそのまま表示するが、単勝/複勝は的中/不的中ではなく「-」にする
    assert "除外" in html_content
    assert html_content.count('<span class="hit-badge void">-</span>') == 2
    assert "的中</span>" not in html_content
    assert "不的中</span>" not in html_content
    # レース名には第何Rかを含む。テーブルにソート機能（class="sortable"）は付けない
    assert "<td>11R テストS</td>" in html_content
    assert 'class="sortable"' not in html_content
    # 日付には曜日（土曜=青）を付ける
    assert '<span class="weekday-sat">(土)</span>' in html_content
    # 除外のみの開催日は的中率・回収率を計算できない旨を表示する
    assert "的中率・回収率を計算できるレースがありません。" in html_content


def test_meeting_race_detail_table_html_shows_day_mini_stats_and_blank_for_missing_values():
    days = [{
        "race_day": date(2026, 6, 14),  # 日曜
        "races": [
            {
                "race_id": "202605030401", "race_name": "テストA", "race_type": "芝", "course_len": "1800",
                "ground_state": "良", "class": "未勝利",
                "winner_name": "ホースA", "pick_name": "ホースA", "pick_pop": "1", "pick_finish": "1",
                "pick_scratched": False, "win_hit": True, "win_payout": 150.0,
                "place_hit": True, "place_payout": 110.0,
            },
            {
                "race_id": "202605030402", "race_name": "テストB",
                # race_result側のデータが欠けているレース（NaN/None混在）
                "race_type": float("nan"), "course_len": None, "ground_state": float("nan"), "class": None,
                "winner_name": None, "pick_name": "ホースC", "pick_pop": None, "pick_finish": None,
                "pick_scratched": False, "win_hit": False, "win_payout": None,
                "place_hit": False, "place_payout": None,
            },
        ],
    }]

    html_content = r._meeting_race_detail_table_html(days)

    print(f"\n--- _meeting_race_detail_table_html（日次ミニ集計・NaN欠損）---\n{html_content}")

    # 日曜は赤
    assert '<span class="weekday-sun">(日)</span>' in html_content
    # 1日分の単勝・複勝それぞれの的中率・回収率（小さく1行、2件中1件的中なので的中率50%）
    assert '<p class="day-mini-stats">単勝 的中率50.0% 回収率75.0% / 複勝 的中率50.0% 回収率55.0% (対象2件)</p>' in html_content
    # NaN/Noneの列は「nan」ではなく「-」で表示する
    assert "nan" not in html_content
    assert "<td>-</td>" in html_content
    assert '<td style="background-color:#ffffff;">-</td>' in html_content


def test_make_all_meeting_performance_pages_generates_for_every_meeting(new_roots, fake_dataset):
    r.make_all_meeting_performance_pages()

    meeting_dir = new_roots / "public_html" / "performance" / "meeting"
    # SAMPLE_DFの4行は (2025,東京,1) (2026,東京,2) (2026,東京,3) (2024,阪神,1) の4開催
    expected_files = [
        meeting_dir / "2025" / "05_tokyo-1th.html",
        meeting_dir / "2026" / "05_tokyo-2th.html",
        meeting_dir / "2026" / "05_tokyo-3th.html",
        meeting_dir / "2024" / "09_hanshin-1th.html",
    ]

    print(f"\n--- make_all_meeting_performance_pages() ---")
    for f in expected_files:
        print(f"  {f}: exists={f.exists()}")

    for f in expected_files:
        assert f.exists()


def test_make_all_meeting_performance_pages_noop_on_empty_dataset(new_roots, empty_dataset):
    r.make_all_meeting_performance_pages()

    meeting_dir = new_roots / "public_html" / "performance" / "meeting"
    assert not meeting_dir.exists()


def test_make_course_performance_page_generates_html(new_roots, fake_dataset):
    r.make_course_performance_page(place_id=5, race_type="芝", course_len="1400")

    out_file = new_roots / "public_html" / "performance" / "course" / "05_tokyo" / "芝-1400.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_course_performance_page(東京, 芝1400m) ---")
    print(html_content)

    assert '<h1>東京 <span class="race-type-turf">芝1400m</span> AI予想成績</h1>' in html_content
    # 東京 芝1400mはA・Bの2レース: win hit=1/2=50.0%, return=(200+0)/2=100.0
    assert return_rate_big_html(100.0) in html_content
    assert hit_rate_gauge_html(50.0) in html_content
    # 内訳テーブルの各セルは、トータル成績カードと同じ「上に回収率(大)・下に的中率(小)」
    assert "<th>単勝</th>" in html_content
    assert '<div class="cell-stat">' in html_content
    assert '<div class="cell-stat-primary">' in html_content
    assert '<div class="cell-stat-primary-gauge">' in html_content
    assert '<div class="cell-stat-secondary">' in html_content
    assert "<h3>クラス別成績</h3>" in html_content
    assert "<h3>馬場別成績</h3>" in html_content
    assert "<h3>年度別成績</h3>" in html_content
    assert "<td>未勝利</td>" in html_content
    assert "<td>1勝クラス</td>" in html_content
    # トータル/今年・クラス別・馬場別・年度別がタブに分かれている
    assert '<div class="tabbed-section">' in html_content
    assert '<button data-target="overview" aria-selected="true">トータル/今年</button>' in html_content
    assert '<button data-target="breakdown" aria-selected="false">クラス別・馬場別</button>' in html_content
    assert '<button data-target="year" aria-selected="false">年度別</button>' in html_content
    assert '<button data-target="cross" aria-selected="false">馬場×クラス</button>' in html_content
    assert '<div class="section-panel" data-section="year" hidden>' in html_content
    assert '<nav class="site-nav">' in html_content
    assert '<script src="../../../assets/js/sortable-table.js"></script>' in html_content
    assert '<script src="../../../assets/js/section-tabs.js"></script>' in html_content
    assert '<script src="../../../assets/js/cross-filter.js"></script>' in html_content
    # 年度別タブに推移グラフ（2026年と2025年のトレンド）が追加されている
    assert '<div class="section-panel" data-section="cross" hidden>' in html_content
    assert '<div class="trend-section">' in html_content
    assert '<svg class="trend-chart"' in html_content
    # 馬場×クラスの絞り込みUI: 馬場/クラスのセレクトと、組み合わせごとの事前計算パネル
    assert '<div class="cross-filter">' in html_content
    assert 'class="cross-filter-ground-state"' in html_content
    assert 'class="cross-filter-class"' in html_content
    assert '<option value="良">良</option>' in html_content
    assert '<option value="稍重">稍重</option>' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="all" data-class="all" hidden>' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="良" data-class="未勝利" hidden>' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="稍重" data-class="1勝クラス" hidden>' in html_content
    # ブレッドクラムが追加されている
    assert '<a href="../../../performance/course/05_tokyo/index.html">東京</a>' in html_content
    assert '<span class="breadcrumb-current"><span class="race-type-turf">芝1400m</span></span>' in html_content
    # 旧来の右サイドバー（競馬場・コース選択）は、右側タブの階層表示と重複するため廃止した
    assert '<aside class="page-sidebar">' not in html_content
    # 右側タブに同じ階層（他の競馬場・このコースの他の距離）が表示される
    assert '<a href="../../../performance/course/06_nakayama/index.html">中山</a>' in html_content
    assert '<a href="../../../performance/course/05_tokyo/ダート-1600.html"><span class="race-type-dirt">ダート1600m</span></a>' in html_content
    assert '<span class="page-calendar-tab-current"><span class="race-type-turf">芝1400m</span></span>' in html_content


def test_make_course_performance_index_page_generates_html(new_roots, fake_dataset):
    r.make_course_performance_index_page(place_id=5)

    out_file = new_roots / "public_html" / "performance" / "course" / "05_tokyo" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_course_performance_index_page(place_id=5) ---")
    print(html_content)

    # AI成績のページ全体に専用のサブカラー（section-performance）を適用する
    assert '<body class="section-performance">' in html_content
    assert "<h1>東京 AI予想成績</h1>" in html_content
    assert '<a href="芝-1400.html"><span class="race-type-turf">芝1400m</span></a>' in html_content
    assert '<a href="../../index.html">&larr; AI成績トップへ</a>' in html_content

    # 東京全体（A・B・C）: win hit=2/3=66.7%, return=(200+0+100)/3=100.0
    assert return_rate_big_html(100.0) in html_content
    assert hit_rate_gauge_html(2 / 3 * 100) in html_content
    # 今年の成績の下に「開催別成績」（新しい開催が上、詳細ページへのリンク付き）が追加されている
    # 東京は2025年第1回(A)・2026年第2回(B)・2026年第3回(C)の3開催
    assert "<h3>開催別成績</h3>" in html_content
    assert html_content.index('<a href="../../meeting/2026/05_tokyo-3th.html">2026年 第3回</a>') < html_content.index(
        '<a href="../../meeting/2026/05_tokyo-2th.html">2026年 第2回</a>'
    )
    assert '<a href="../../meeting/2025/05_tokyo-1th.html">2025年 第1回</a>' in html_content

    # 今年（2026年）の成績の下に、直近の開催週別の傾向・推移が追加されている
    # （B=2026-04-01週、C=2026-04-08週は7日違いの同じ曜日なので別の週開始日になる）
    # （年度別タブの年次トレンドと合わせて、推移グラフは単勝/複勝×2セクション=4つになる）
    assert "<h4>直近の開催週別の傾向・推移</h4>" in html_content
    assert html_content.count('<svg class="trend-chart"') == 4
    assert ">3/28</text>" in html_content
    assert ">4/4</text>" in html_content
    assert "<h3>開催週別成績</h3>" in html_content
    assert "<td>2026-03-28</td>" in html_content
    assert "<td>2026-04-04</td>" in html_content
    assert "<h3>年度別成績</h3>" in html_content
    assert "<h3>クラス別成績</h3>" in html_content
    assert "<h3>芝/ダート別成績</h3>" in html_content
    assert "<h3>馬場別成績</h3>" in html_content
    # 内訳テーブルの回収率列は概要カードと同じ「大きな数字+ゲージ」で表示する
    assert '<div class="cell-stat-primary">' in html_content
    assert '<div class="cell-stat-primary-gauge">' in html_content
    # トータル/今年・クラス別等・年度別がタブに分かれている
    assert '<div class="tabbed-section">' in html_content
    assert '<button data-target="breakdown" aria-selected="false">クラス別・芝ダート別・馬場別</button>' in html_content
    assert '<button data-target="cross" aria-selected="false">馬場×クラス</button>' in html_content
    assert '<div class="section-panel" data-section="year" hidden>' in html_content
    assert '<script src="../../../assets/js/section-tabs.js"></script>' in html_content
    assert '<script src="../../../assets/js/cross-filter.js"></script>' in html_content
    assert '<nav class="site-nav">' in html_content
    # 年度別タブに推移グラフ（年次トレンド）が追加されている
    assert '<div class="trend-section">' in html_content
    assert '<svg class="trend-chart"' in html_content
    # 馬場×クラスの絞り込みUI: 東京全体（A・C=良×未勝利、B=稍重×1勝クラス）の組み合わせパネル
    assert '<div class="cross-filter">' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="all" data-class="all" hidden>' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="良" data-class="未勝利" hidden>' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="稍重" data-class="1勝クラス" hidden>' in html_content
    # ブレッドクラムが追加されている
    assert '<a href="../../../performance/index.html">AI成績</a>' in html_content
    assert '<span class="breadcrumb-current">東京</span>' in html_content
    # 旧来の右サイドバー（競馬場選択）は、右側タブの階層表示と重複するため廃止した
    assert '<aside class="page-sidebar">' not in html_content
    # 右側タブに同じ階層（他の競馬場・東京の全コース）が表示される
    assert '<a href="../../../performance/course/06_nakayama/index.html">中山</a>' in html_content
    assert '<span class="page-calendar-tab-current">東京</span>' in html_content


def test_make_all_annual_performance_pages_generates_for_each_year(new_roots, fake_dataset):
    r.make_all_annual_performance_pages()

    assert (new_roots / "public_html" / "performance" / "annual" / "2024.html").exists()
    assert (new_roots / "public_html" / "performance" / "annual" / "2025.html").exists()
    assert (new_roots / "public_html" / "performance" / "annual" / "2026.html").exists()


def test_make_all_course_performance_pages_generates_for_every_place(new_roots, fake_dataset):
    r.make_all_course_performance_pages()

    out_dir = new_roots / "public_html" / "performance" / "course"
    place_dirs = sorted(p.name for p in out_dir.iterdir())

    print(f"\n--- make_all_course_performance_pages() ---")
    print(f"  生成された開催場ディレクトリ: {place_dirs}")

    assert len(place_dirs) == 10
    # 05_tokyoには index.html とコース別ページが両方生成される
    tokyo_files = sorted(p.name for p in (out_dir / "05_tokyo").iterdir())
    assert "index.html" in tokyo_files
    assert "芝-1400.html" in tokyo_files
