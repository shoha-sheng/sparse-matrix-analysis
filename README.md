# 疎行列の二乗計算プログラム

本プロジェクトは、疎行列の二乗計算を行うためのプログラムである。  
3種類のアルゴリズム（Naïve, ハッシュベースSpGEMM, GustavsonベースSpGEMM）を用い、異なるデータ構造に適用することで計算速度の比較を行う。

## 📂 プロジェクト構成

```
📄 kadai.c              # 疎行列の二乗計算プログラム（C言語）
📄 run_experiments.py    # 実験の自動実行スクリプト（Python）
📄 visualize_results.py  # 実行結果の可視化スクリプト（Python）
📄 README.md             # 本ファイル
```

## 🛠 使用方法

### **1️⃣ Cプログラムのコンパイル**
以下のコマンドを実行し、Cプログラムをコンパイルする。
```bash
gcc kadai.c -O3 -std=c99 -o a.out
```

### **2️⃣ 実験の実行**
Pythonスクリプトを実行し、異なるアルゴリズムとデータ構造の組み合わせで計算を行い、その結果をCSVファイルに記録する。
```bash
python run_experiments.py
```
実行が完了すると、`execution_results.csv` というファイルが生成される。

### **3️⃣ 実行結果の可視化**
記録された実行時間を可視化し、各アルゴリズムの特性を分析する。
```bash
python visualize_results.py
```
このスクリプトの実行により、以下のグラフ画像が出力される。

## 📊 生成される可視化データ
可視化スクリプト `visualize_results.py` を実行すると、以下の画像ファイルが作成される。

- **`boxplot.png`**  
  各データ構造ごとの実行時間の分布を示す箱ひげ図。
- **`trend_algorithm.png`**  
  行列サイズごとのアルゴリズムの実行時間推移を示す折れ線グラフ。
- **`trend_structure.png`**  
  行列サイズごとのデータ構造の比較を示す折れ線グラフ。
- **`heatmap.png`**  
  データ構造×アルゴリズムの平均実行時間を示すヒートマップ。

これらの可視化により、最適なアルゴリズムとデータ構造の組み合わせを分析できる。

## 🏆 最速のアルゴリズムとデータ構造
実験結果から、各行列サイズごとに最速の **（データ構造 × アルゴリズム）** の組み合わせを抽出し、表形式で記録する。
