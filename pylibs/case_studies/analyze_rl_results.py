import pandas as pd
import matplotlib.pyplot as plt

def main():
    # CSV 데이터 로드
    csv_path = "tmp/br_reed_bandit_rl.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {csv_path}")
        return

    # 시간 단위로 그룹화하여 네트워크 전체 평균 및 합계 계산
    # (결과가 너무 빽빽하지 않도록 10개의 데이터 단위로 이동 평균 적용)
    df_grouped = df.groupby('sim_time_s').agg({
        'reward': 'mean',
        'app_drops_interval': 'sum',
        'cca_fails_interval': 'sum'
    }).reset_index()
    
    window_size = max(1, len(df_grouped) // 50) # 부드러운 그래프를 위한 윈도우 사이즈
    
    plt.figure(figsize=(14, 10))
    
    # 1. 평균 보상(Reward) 변화
    plt.subplot(3, 1, 1)
    plt.plot(df_grouped['sim_time_s'], df_grouped['reward'].rolling(window=window_size).mean(), color='blue')
    plt.title("Average Reward over Time (Higher is better)")
    plt.ylabel("Reward")
    
    # 2. 패킷 버림(App Drops) 변화
    plt.subplot(3, 1, 2)
    plt.plot(df_grouped['sim_time_s'], df_grouped['app_drops_interval'].rolling(window=window_size).mean(), color='red')
    plt.title("Total App Drops over Time")
    plt.ylabel("Drop Count")
    
    # 3. 무선 충돌(CCA Fails) 변화
    plt.subplot(3, 1, 3)
    plt.plot(df_grouped['sim_time_s'], df_grouped['cca_fails_interval'].rolling(window=window_size).mean(), color='orange')
    plt.title("Total CCA Fails over Time")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("CCA Fail Count")
    
    plt.tight_layout()
    plt.savefig("tmp/rl_analysis_plot.png")
    print("그래프가 tmp/rl_analysis_plot.png 에 저장되었습니다.")

if __name__ == "__main__":
    main()
