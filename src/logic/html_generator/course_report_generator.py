"""コース詳細データページ（Forge: HTMLFactory）のHTML生成

各開催場・コース（芝/ダート×距離）について、race_info_dataset_manager /
peds_results_dataset_manager の週次更新で既に集計済みのコース別データ
（平均勝ち時計・平均人気・勝ち馬の平均馬体重・枠番馬番傾向・血統別成績）を
ページとして表示する。集計ロジック自体は新規実装しない（既存データを読むのみ）。

旧ver2.0の web/site/performance/course/ 相当（ただしver2.0でも的中率・回収率以外の
コース詳細データ自体は専用ページとして仕上がっていなかった）。
"""

from src.config.constants import NAME_LIST, PLACE_LIST
from src.config.lists import COURSE_LISTS
from src.managers import html_manager, peds_results_dataset_manager, race_info_dataset_manager


def _filter_overall_row(df, race_type, course_len, ground_state_label="全"):
    """集計CSV(df)から、指定race_type/course_lenの「全体(ground_state=全, class=all)」行を1件取り出す"""
    if df.empty:
        return None
    cond = (
        (df["race_type"] == race_type)
        & (df["course_len"].astype(str) == str(course_len))
        & (df["ground_state"] == ground_state_label)
        & (df["class"] == "all")
    )
    sub = df[cond]
    if sub.empty:
        return None
    return sub.iloc[0]


def build_course_report(place_id, race_type, course_len):
    """1コース条件（開催場×race_type×距離）分のレポートデータを辞書で返す"""
    avg_time = _filter_overall_row(
        race_info_dataset_manager.get_total_average_time_csv(place_id), race_type, course_len
    )
    avg_pop = _filter_overall_row(
        race_info_dataset_manager.get_total_average_pops_csv(place_id), race_type, course_len
    )
    winner_weight = _filter_overall_row(
        race_info_dataset_manager.get_total_winner_weight_csv(place_id), race_type, course_len
    )
    avg_frame_and_horse = _filter_overall_row(
        race_info_dataset_manager.get_total_average_frames_csv(place_id), race_type, course_len
    )
    peds_df = peds_results_dataset_manager.get_total_peds_results_csv(place_id, race_type, course_len, "all")

    return {
        "place_id": place_id,
        "race_type": race_type,
        "course_len": course_len,
        "avg_time": avg_time,
        "avg_pop": avg_pop,
        "winner_weight": winner_weight,
        "avg_frame_and_horse": avg_frame_and_horse,
        "peds_df": peds_df,
    }


def _fmt(row, key, suffix=""):
    if row is None or key not in row.index:
        return "データなし"
    value = row[key]
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return "データなし"
    return f"{value}{suffix}"


def course_report_to_html(report):
    place_name = NAME_LIST[report["place_id"] - 1]
    race_type = report["race_type"]
    course_len = report["course_len"]

    avg_time_html = _fmt(report["avg_time"], "avg_time", "ms")
    avg_pop_html = _fmt(report["avg_pop"], "avg_pop")
    winner_weight_html = _fmt(report["winner_weight"], "馬体重", "kg")
    avg_frame_html = _fmt(report["avg_frame_and_horse"], "avg_frame")
    avg_horse_html = _fmt(report["avg_frame_and_horse"], "avg_horse")

    peds_df = report["peds_df"]
    if not peds_df.empty:
        peds_rows = "".join(
            f"<tr><td>{row['血統']}</td><td>{row['1着']}</td><td>{row['2着']}</td>"
            f"<td>{row['3着']}</td><td>{row['着外']}</td></tr>\n"
            for _, row in peds_df.head(10).iterrows()
        )
    else:
        peds_rows = "<tr><td colspan='5'>データなし</td></tr>"

    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{place_name} {race_type}{course_len}m コース詳細</title>
  <link rel="stylesheet" href="../../assets/css/styles.css">
</head>
<body>
  <p><a href="index.html">&larr; {place_name}のコース一覧へ</a></p>
  <h1>{place_name} {race_type}{course_len}m コース詳細</h1>

  <h2>コース別平均データ</h2>
  <ul>
    <li>平均勝ち時計: {avg_time_html}</li>
    <li>平均人気: {avg_pop_html}</li>
    <li>勝ち馬の平均馬体重: {winner_weight_html}</li>
    <li>勝ち馬の平均枠番: {avg_frame_html} / 平均馬番: {avg_horse_html}</li>
  </ul>

  <h2>血統別成績（上位10件）</h2>
  <table>
    <thead><tr><th>血統(父)</th><th>1着</th><th>2着</th><th>3着</th><th>着外</th></tr></thead>
    <tbody>
      {peds_rows}
    </tbody>
  </table>

  <p><a href="../../index.html">&larr; HOMEへ戻る</a></p>
</body>
</html>
"""


def make_course_index_page():
    """全開催場一覧ページ（public_html/courses/index.html）を生成する"""
    rows = "".join(
        f'<li><a href="{PLACE_LIST[i]}/index.html">{NAME_LIST[i]}</a></li>\n' for i in range(len(PLACE_LIST))
    )
    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>コース詳細データ</title>
  <link rel="stylesheet" href="../assets/css/styles.css">
</head>
<body>
  <h1>コース詳細データ</h1>
  <ul>
    {rows}
  </ul>
  <p><a href="../index.html">&larr; HOMEへ戻る</a></p>
</body>
</html>
"""
    html_manager.save_course_index_html(html)


def make_track_page(place_id):
    """場別のコース×距離一覧ページ（public_html/courses/{place}/index.html）を生成する"""
    place_name = NAME_LIST[place_id - 1]
    course_list = COURSE_LISTS[place_id - 1]
    rows = "".join(
        f'<li><a href="{race_type}-{course_len}.html">{race_type}{course_len}m</a></li>\n'
        for race_type, course_len in course_list
    )
    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{place_name} コース一覧</title>
  <link rel="stylesheet" href="../../assets/css/styles.css">
</head>
<body>
  <h1>{place_name} コース一覧</h1>
  <ul>
    {rows}
  </ul>
  <p><a href="../index.html">&larr; コース詳細データ一覧へ</a></p>
</body>
</html>
"""
    html_manager.save_track_index_html(place_id, html)


def make_course_detail_page(place_id, race_type, course_len):
    """個別コース詳細ページ（public_html/courses/{place}/{race_type}-{course_len}.html）を生成する"""
    report = build_course_report(place_id, race_type, course_len)
    html_manager.save_course_detail_html(place_id, race_type, course_len, course_report_to_html(report))


def make_all_course_pages():
    """全開催場・全コースのコース詳細データページを一括生成する"""
    make_course_index_page()
    for place_id in range(1, len(PLACE_LIST) + 1):
        make_track_page(place_id)
        for race_type, course_len in COURSE_LISTS[place_id - 1]:
            make_course_detail_page(place_id, race_type, course_len)
