import sys
sys.path.append(r"C:\\keiba_ai\\keiba_ai_ver2.0\\src\\Datasets")
sys.dont_write_bytecode = True
import threading
import time

import analysis_race_info
import analysis_race_time
import average_time
import race_results
import race_returns
import horse_peds
import peds_results
import past_performance

sys.path.append(r"C:\\keiba_ai\\keiba_ai_ver2.0\\src\\PredictionModels")
import LightGBM.make_dataset  as LightGBM_dataset

def timer(timeout, event):
    time.sleep(timeout)
    if not event.is_set():
        event.set()

def wait_for_enter(event):
    input("Enterキーを押してください...")
    if not event.is_set():
        print("\nEnterキーが押されました。")
        event.set()

if __name__ == "__main__":  
    # データセットの更新
    race_results.weekly_update_race_results()
    horse_peds.weekly_update_horse_peds()
    peds_results.weekly_update_pedsdata()
    # # race_returns.weekly_update_race_returns()
    average_time.timedata_update()
    analysis_race_time.weekly_winners_time_update()
    analysis_race_info.weekly_update_average_pops()
    analysis_race_info.weekly_update_winners_weight()
    analysis_race_info.weekly_update_average_frame_and_horse()
    analysis_race_info.weekly_update_horse_name_id_map()
    past_performance.weekly_update_past_performance()
    
    # 予想スコアの計算
    LightGBM_dataset.weekly_update_dataset_for_train()

    print("Weekly Update Done")
    # 3分（180秒）設定
    timeout = 180
    event = threading.Event()

    # スレッド作成
    timer_thread = threading.Thread(target=timer, args=(timeout, event))
    input_thread = threading.Thread(target=wait_for_enter, args=(event,))

    # スレッドをデーモンスレッドとして設定
    timer_thread.daemon = True
    input_thread.daemon = True

    # スレッド開始
    timer_thread.start()
    input_thread.start()

    # イベントがセットされるのを待機
    event.wait()
    
    