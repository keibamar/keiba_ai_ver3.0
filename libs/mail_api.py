import smtplib
import ssl
from email.mime.text import MIMEText

import name_header

#ご自身のgmailアドレス
account = "keibamar18@gmail.com"
#アプリパスワード（１６桁）
password = "zgor pdzu aaun pwzj"

# 送信先のアドレスを登録します
send_address = "taired999@gmail.com"

# メインの関数
def send_email(file_path):
    # txtファイルを読み込む
    body = read_txt_file(file_path)
    title, body = process_txt_file(file_path)
    
    # ファイルが見つからなかった場合は処理を終了
    if title == "エラー":
        print("メール送信を中止しました。")
        return
    
    # メールを作成して送信
    msg = make_mime_text(
        mail_to=send_address,
        subject=title,
        body=body
    )
    send_gmail(msg)
    print("メール送信成功！:", title)

# txtファイルの内容を読み込む関数（改行を考慮）
def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print("指定されたファイルが見つかりません！")
        return ""
    
# txtファイルの内容を処理し、タイトルと本文を返す関数
def process_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]  # 空行を削除
            if len(lines) < 2:
                print("ファイルに十分な内容がありません！")
                return "タイトルなし", "\n".join(lines)
            title = lines[0] + " " + lines[1]  # 最初の2行をタイトルに結合
            body = "\n".join(lines[2:])  # 3行目以降を本文に
            return title, body
    except FileNotFoundError:
        print("指定されたファイルが見つかりません！")
        return "エラー", "ファイルが見つかりませんでした！"
    
# 件名、送信先アドレス、本文を渡してメールオブジェクトを生成する関数
def make_mime_text(mail_to, subject, body):
    # MIMETextで改行を保持
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["To"] = mail_to
    msg["From"] = account
    return msg

# smtp経由でメール送信する関数
def send_gmail(msg):
    try:
        server = smtplib.SMTP_SSL(
            "smtp.gmail.com", 465,
            context=ssl.create_default_context()
        )
        server.set_debuglevel(0)
        server.login(account, password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"メール送信中にエラーが発生しました: {e}")

def send_race_pred(race_day, race_id):
    text_path = name_header.TEXT_PATH + "race_prediction/" + race_day.strftime("%Y%m%d") + "/" + str(race_id) + ".txt"
    send_email(text_path)

# 実行部分
if __name__ == "__main__":
    file_path = "sample.txt"
    send_email(file_path)