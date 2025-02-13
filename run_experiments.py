import subprocess
import time
import csv

# 設定: 実験対象の行列・データ構造・アルゴリズム
matrices = [f"bcspwr{i:02d}.txt" for i in range(1, 11)]
data_structures = ["dense", "csr", "coo", "list", "skyline"]
algorithms = ["naive", "hash", "gustavson"]

# 出力CSVファイル名
output_csv = "execution_results.csv"

# ヘッダー情報
header = ["Matrix", "Data Structure", "Algorithm", "Execution Time (s)"]

# 結果を格納するリスト
results = []

# 実験実行
for matrix in matrices:
    for ds in data_structures:
        for alg in algorithms:
            cmd = f"./a.out -d {ds} -a {alg} < {matrix}"
            print(f"Running: {cmd}")

            # 実行時間を測定
            start_time = time.perf_counter()
            process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            end_time = time.perf_counter()

            # 実行時間（秒）
            exec_time = end_time - start_time

            # 実行結果をリストに追加
            results.append([matrix, ds, alg, exec_time])

# CSVに保存
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(results)

print(f"実行結果を {output_csv} に保存しました。")