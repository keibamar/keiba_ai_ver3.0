import re

def format_error(date_string):
    print(date_string, "はフォーマットで無効です。")
    print("指定されたフォーマットで入力指定ください。")
    return

def is_valid_date_format(date_string):
    """有効な日付のフォーマットになっているかチェックする
        Args:
            date_string(str) : 文字列
        Return:
            True / False
    """
    # 正規表現でフォーマットチェック
    pattern = re.compile(r"^\d{4}\d{2}\d{2}$")
    if pattern.match(date_string):
        try:
            # 年、月、日を抽出
            year = int(date_string[:4])
            month = int(date_string[4:6])
            day = int(date_string[6:8])
            # 月と日が範囲内にあるか確認
            if not (1 <= month <= 12):
                print("月が無効です。")
                return False
            if not (1 <= day <= 31):
                print("日が無効です。")
                return False
            return True
        except ValueError:
            print(date_string, "を数値に変換できません。")
            return False
    else:
        # format_error(date_string)
        return False