cd c:\keiba_ai\keiba_ai_ver3.0
python scripts\make_v9_datasets_no26.py >> logs\make_v9_datasets_log.txt 2>> logs\make_v9_datasets_err.txt
if ($LASTEXITCODE -eq 0) {
    echo "[AUTO] v9完了 -> train_v15pace_no26.py 開始" >> logs\train_v15pace_log.txt
    python scripts\train_v15pace_no26.py >> logs\train_v15pace_log.txt 2>> logs\train_v15pace_err.txt
    echo "[AUTO] train完了" >> logs\train_v15pace_log.txt
} else {
    echo "[AUTO] v9でエラー発生、trainはスキップ" >> logs\train_v15pace_log.txt
}
