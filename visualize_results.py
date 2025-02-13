import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 日本語フォントのパスを指定（macOSの例）
font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
font_prop = fm.FontProperties(fname=font_path)

# matplotlib にフォントを適用
plt.rcParams["font.family"] = font_prop.get_name()

# 1️⃣ CSVを読み込む
df = pd.read_csv("execution_results.csv")

# 2️⃣ データの確認
print(df.head())  # 先頭5行を表示

# 3️⃣ 箱ひげ図（データ構造ごとの実行時間の分布）
plt.figure(figsize=(12, 6))
sns.boxplot(x="Data Structure", y="Execution Time (s)", hue="Algorithm", data=df)
plt.yscale("log")  # 実行時間のスケールをログ変換
plt.title("データ構造ごとの実行時間の比較", fontproperties=font_prop)
plt.xlabel("データ構造", fontproperties=font_prop)
plt.ylabel("実行時間（秒）", fontproperties=font_prop)
plt.legend(title="アルゴリズム", prop=font_prop)
plt.savefig("boxplot.png")
plt.show()

# 4️⃣ 実行時間のトレンド（アルゴリズムごと）
plt.figure(figsize=(12, 6))
for algo in df["Algorithm"].unique():
    subset = df[df["Algorithm"] == algo]
    avg_time = subset.groupby("Matrix")["Execution Time (s)"].mean()
    plt.plot(avg_time.index, avg_time.values, label=algo, marker="o")

plt.yscale("log")
plt.xlabel("行列サイズ (Matrix)", fontproperties=font_prop)
plt.ylabel("実行時間（秒）", fontproperties=font_prop)
plt.title("アルゴリズムごとの実行時間推移", fontproperties=font_prop)
plt.xticks(rotation=45)
plt.legend(title="アルゴリズム", prop=font_prop)
plt.grid(True)
plt.savefig("trend_algorithm.png")
plt.show()

# 5️⃣ 実行時間のトレンド（データ構造ごと）
plt.figure(figsize=(12, 6))
for ds in df["Data Structure"].unique():
    subset = df[df["Data Structure"] == ds]
    avg_time = subset.groupby("Matrix")["Execution Time (s)"].mean()
    plt.plot(avg_time.index, avg_time.values, label=ds, marker="o")

plt.yscale("log")
plt.xlabel("行列サイズ (Matrix)", fontproperties=font_prop)
plt.ylabel("実行時間（秒）", fontproperties=font_prop)
plt.title("データ構造ごとの実行時間推移", fontproperties=font_prop)
plt.xticks(rotation=45)
plt.legend(title="データ構造", prop=font_prop)
plt.grid(True)
plt.savefig("trend_structure.png")
plt.show()

# 6️⃣ ヒートマップ（データ構造 × アルゴリズムの平均実行時間）
plt.figure(figsize=(10, 6))
heatmap_data = df.pivot_table(index="Data Structure", columns="Algorithm", values="Execution Time (s)", aggfunc="mean")
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5)

plt.title("データ構造 × アルゴリズムの平均実行時間", fontproperties=font_prop)
plt.xlabel("アルゴリズム", fontproperties=font_prop)
plt.ylabel("データ構造", fontproperties=font_prop)
plt.savefig("heatmap.png")
plt.show()

# 各行列ごとに最速のデータ構造とアルゴリズムを特定
fastest_per_matrix = df.loc[df.groupby("Matrix")["Execution Time (s)"].idxmin()]

# 出力を整理
fastest_per_matrix = fastest_per_matrix.sort_values(by="Matrix").reset_index(drop=True)

# 表として表示
print("Fastest Execution Results")
print(fastest_per_matrix)