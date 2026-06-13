"""race_schedule（race_calendar）データセットの検証ロジック"""

from src.datasets.race_schedule import model


def has_calendar_columns(calendar_df):
    """calendar_dfがrace_calendarとして必要な列を全て持っているか確認する"""
    return all(column in calendar_df.columns for column in model.CALENDAR_COLUMNS)
