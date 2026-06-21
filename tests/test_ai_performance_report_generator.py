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

import pandas as pd
import pytest

from src.config import paths
from src.logic.html_generator import ai_performance_report_generator as r
from src.logic.html_generator.rate_gauge_html import hit_rate_gauge_html, return_rate_gauge_html

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
    assert (
        f"<td>単勝</td><td>{hit_rate_gauge_html(50.0)}</td><td>{return_rate_gauge_html(75.0)}</td><td>4</td>"
        in html_content
    )
    # 新しい年から順にリンクされる
    assert html_content.index("annual/2026.html") < html_content.index("annual/2025.html") < html_content.index("annual/2024.html")
    assert '<a href="annual/2026.html">2026年</a>' in html_content
    assert '<a href="course/05_tokyo/index.html">東京</a>' in html_content
    assert '<nav class="site-nav">' in html_content
    assert '<table class="sortable">' in html_content
    assert '<script src="../assets/js/sortable-table.js"></script>' in html_content
    # トップ階層はブレッドクラムのみ（サイドバーは追加しない）
    assert '<p class="breadcrumb">' in html_content
    assert '<span class="breadcrumb-current">AI成績</span>' in html_content
    assert '<aside class="page-sidebar">' not in html_content
    # 年間成績の上に、単勝的中率・回収率の年次トレンド（スパークライン）が追加されている
    assert '<div class="trend-section">' in html_content
    assert '<svg class="sparkline"' in html_content
    assert html_content.count('<svg class="sparkline"') == 2


def test_make_ai_performance_index_page_handles_empty_dataset(new_roots, empty_dataset):
    r.make_ai_performance_index_page()

    out_file = new_roots / "public_html" / "performance" / "index.html"
    html_content = out_file.read_text(encoding="utf-8")
    assert "予想データがまだありません。" in html_content
    assert '<svg class="sparkline"' not in html_content


def test_make_annual_performance_page_generates_html(new_roots, fake_dataset):
    r.make_annual_performance_page(2026)

    out_file = new_roots / "public_html" / "performance" / "annual" / "2026.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_annual_performance_page(2026) ---")
    print(html_content)

    assert "<h1>2026年 AI予想成績</h1>" in html_content
    # 2026年はB・Cの2レース: win hit=1/2=50.0%, return=(0+100)/2=50.0
    assert (
        f"<td>単勝</td><td>{hit_rate_gauge_html(50.0)}</td><td>{return_rate_gauge_html(50.0)}</td><td>2</td>"
        in html_content
    )
    # ブレッドクラムと、他の年度へのサイドバーが追加されている
    assert '<a href="../../performance/index.html">AI成績</a>' in html_content
    assert '<span class="breadcrumb-current">2026年</span>' in html_content
    assert '<aside class="page-sidebar">' in html_content
    assert '<p class="page-sidebar-up"><a href="../index.html">&uarr; AI成績トップ</a></p>' in html_content
    assert "<h3>年度</h3>" in html_content
    assert '<li><a href="2025.html">2025年</a></li>' in html_content
    assert '<li><span class="page-sidebar-current">2026年</span></li>' in html_content


def test_make_meeting_performance_page_generates_html(new_roots, fake_dataset):
    r.make_meeting_performance_page(2025, place_id=5, times=1)

    out_file = new_roots / "public_html" / "performance" / "meeting" / "2025" / "05_tokyo-1th.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")
    assert "<h1>2025年 東京1回 AI予想成績</h1>" in html_content
    # 2025年東京1回はAのみ: win hit=1/1=100.0%, return=200.0
    assert (
        f"<td>単勝</td><td>{hit_rate_gauge_html(100.0)}</td><td>{return_rate_gauge_html(200.0)}</td><td>1</td>"
        in html_content
    )
    # 開催別ページはブレッドクラムのみ（サイドバーは追加しない）
    assert '<p class="breadcrumb">' in html_content
    assert '<span class="breadcrumb-current">2025年 東京1回</span>' in html_content
    assert '<aside class="page-sidebar">' not in html_content


def test_make_course_performance_page_generates_html(new_roots, fake_dataset):
    r.make_course_performance_page(place_id=5, race_type="芝", course_len="1400")

    out_file = new_roots / "public_html" / "performance" / "course" / "05_tokyo" / "芝-1400.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_course_performance_page(東京, 芝1400m) ---")
    print(html_content)

    assert "<h1>東京 芝1400m AI予想成績</h1>" in html_content
    # 東京 芝1400mはA・Bの2レース: win hit=1/2=50.0%, return=(200+0)/2=100.0
    assert (
        f"<td>単勝</td><td>{hit_rate_gauge_html(50.0)}</td><td>{return_rate_gauge_html(100.0)}</td><td>2</td>"
        in html_content
    )
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
    # 年度別タブにスパークライン（2026年と2025年のトレンド）が追加されている
    assert '<div class="section-panel" data-section="cross" hidden>' in html_content
    assert '<div class="trend-section">' in html_content
    assert '<svg class="sparkline"' in html_content
    # 馬場×クラスの絞り込みUI: 馬場/クラスのセレクトと、組み合わせごとの事前計算パネル
    assert '<div class="cross-filter">' in html_content
    assert 'class="cross-filter-ground-state"' in html_content
    assert 'class="cross-filter-class"' in html_content
    assert '<option value="良">良</option>' in html_content
    assert '<option value="稍重">稍重</option>' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="all" data-class="all" hidden>' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="良" data-class="未勝利" hidden>' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="稍重" data-class="1勝クラス" hidden>' in html_content
    # ブレッドクラムと、東京の他のコースへのサイドバーが追加されている
    assert '<a href="../../../performance/course/05_tokyo/index.html">東京</a>' in html_content
    assert '<span class="breadcrumb-current">芝1400m</span>' in html_content
    assert '<aside class="page-sidebar">' in html_content
    assert '<p class="page-sidebar-up"><a href="index.html">&uarr; 東京のAI成績</a></p>' in html_content
    assert "<h3>競馬場</h3>" in html_content
    assert '<li><a href="../06_nakayama/index.html">中山</a></li>' in html_content
    assert "<h3>東京のコース</h3>" in html_content
    assert '<li><a href="ダート-1600.html">ダート1600m</a></li>' in html_content
    assert '<li><span class="page-sidebar-current">芝1400m</span></li>' in html_content


def test_make_course_performance_index_page_generates_html(new_roots, fake_dataset):
    r.make_course_performance_index_page(place_id=5)

    out_file = new_roots / "public_html" / "performance" / "course" / "05_tokyo" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_course_performance_index_page(place_id=5) ---")
    print(html_content)

    assert "<h1>東京 AI予想成績</h1>" in html_content
    assert '<a href="芝-1400.html">芝1400m</a>' in html_content
    assert '<a href="../../index.html">&larr; AI成績トップへ</a>' in html_content

    # 東京全体（A・B・C）: win hit=2/3=66.7%, return=(200+0+100)/3=100.0
    assert (
        f"<td>単勝</td><td>{hit_rate_gauge_html(2 / 3 * 100)}</td><td>{return_rate_gauge_html(100.0)}</td><td>3</td>"
        in html_content
    )
    assert "<h3>年度別成績</h3>" in html_content
    assert "<h3>クラス別成績</h3>" in html_content
    assert "<h3>芝/ダート別成績</h3>" in html_content
    assert "<h3>馬場別成績</h3>" in html_content
    # トータル/今年・クラス別等・年度別がタブに分かれている
    assert '<div class="tabbed-section">' in html_content
    assert '<button data-target="breakdown" aria-selected="false">クラス別・芝ダート別・馬場別</button>' in html_content
    assert '<button data-target="cross" aria-selected="false">馬場×クラス</button>' in html_content
    assert '<div class="section-panel" data-section="year" hidden>' in html_content
    assert '<script src="../../../assets/js/section-tabs.js"></script>' in html_content
    assert '<script src="../../../assets/js/cross-filter.js"></script>' in html_content
    assert '<nav class="site-nav">' in html_content
    # 年度別タブにスパークライン（年次トレンド）が追加されている
    assert '<div class="trend-section">' in html_content
    assert '<svg class="sparkline"' in html_content
    # 馬場×クラスの絞り込みUI: 東京全体（A・C=良×未勝利、B=稍重×1勝クラス）の組み合わせパネル
    assert '<div class="cross-filter">' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="all" data-class="all" hidden>' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="良" data-class="未勝利" hidden>' in html_content
    assert '<div class="cross-filter-panel" data-ground-state="稍重" data-class="1勝クラス" hidden>' in html_content
    # ブレッドクラムと、他の競馬場へのサイドバーが追加されている
    assert '<a href="../../../performance/index.html">AI成績</a>' in html_content
    assert '<span class="breadcrumb-current">東京</span>' in html_content
    assert '<aside class="page-sidebar">' in html_content
    assert '<p class="page-sidebar-up"><a href="../../index.html">&uarr; AI成績トップ</a></p>' in html_content
    assert "<h3>競馬場</h3>" in html_content
    assert '<li><a href="../06_nakayama/index.html">中山</a></li>' in html_content
    assert '<li><span class="page-sidebar-current">東京</span></li>' in html_content


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
