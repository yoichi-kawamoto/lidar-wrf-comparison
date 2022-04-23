# 1. 概要
[日本建築学会環境系論文集 第87巻 第797号 pp.460-471（2022年7月）](https://doi.org/10.3130/aije.87.460)で使用したデータとスクリプトを公開しています。

# 2. データとスクリプトのダウンロード
ページ上部の緑色の「Code」をクリックし、「Download ZIP」をクリックします。

## 2.1. ディレクトリ構成
ダウンロードしたzipファイルを展開すると「lidar-wrf-comparison-main」ディレクトリが作成され、その下に以下の構成のディレクトリが作成されます。「3. Taylor Diagramの作成」は「taylor_diagram」ディレクトリ内で、「4. グラフの作成」は「timeseries」ディレクトリ内で実行します。

```
.
├── taylor_diagram
    └── statistics
└── timeseries
    ├── 2015-08-11
    │   ├── Fukuoka
    │   │   ├── sfc
    │   │   │   ├── data
    │   │   │   └── jma_obs
    │   │   └── vert
    │   │       └── data
    │   └── Ohashi
    │       ├── sfc
    │       │   └── data
    │       └── vert
    │           ├── data
    │           └── lidar_obs
    ├── ...
```

# 3. Taylor Diagramの作成
[NCL (NCAR Command Language)](https://www.ncl.ucar.edu/index.shtml) のスクリプトを用いてTaylor Diagramを作成します。スクリプトは以下のコマンドで実行します。

`ncl 2015-08-11_Fukuoka_taylor.ncl`

実行すると、作業ディレクトリ内に画像ファイルが出力されます。リポジトリ内にはpdfファイル形式の出力が含まれています。

## 3.1. taylor_diagram.ncl
Taylor Diagramの設定ファイルです。

## 3.2. 2015-0*-**_Fukuoka_taylor.ncl, 2015-0*-**_Ohashi_taylor.ncl
福岡管区気象台の観測値とWRFの計算値のTaylor Diagramと、Doppler LIDARを用いた高度88 mの観測値とWRFの計算値のTaylor Diagramを作成するためのスクリプトです。Taylor Diagramで使用した統計量は「statistics」ディレクトリ内にあります。

# 4. グラフの作成
Pythonのスクリプトを用いてグラフを作成します。外部パッケージとして、NumPy、pandas、Matplotlibが必要です。以下の環境で動作を確認しています。

- Python: 3.8.12
- NumPy: 1.20.3
- pandas: 1.3.4
- Matplotlib: 3.4.3

フォントはArialを使用しています。Arialがインストールされていない環境ではエラーとなるので、その場合は各スクリプト中の以下の行をコメントアウトするか、他のフォントを指定してください。

```
plt.rcParams["font.family"] = "Arial"
```

スクリプトは以下のコマンドで実行します。

`python plot_sfc_timeseries.py`

または

`python3 plot_sfc_timeseries.py`

なお、「4. グラフの作成」では多数のファイルが作成され、またファイルサイズも大きくなることから、リポジトリ内には出力の画像ファイルは含まれません。

## 4.1. plot_sfc_timeseries.py
地表近傍の気象要素のグラフを作成します。福岡管区気象台を含むメッシュでは、福岡管区気象台の観測値とWRFの計算値の時系列データのグラフを作成します。また、観測値と計算値の誤差等の統計量も併せて出力します。大橋は地表近傍の観測値は無いため、WRFの計算値のみグラフを作成します。グラフは「sfc」の下に作成される「timeseries_plots」ディレクトリに、統計量は「statistics」ディレクトリに出力されます。

## 4.2. plot_vert_timeseries.py
WRFの高度毎にWRFの計算値の時系列データのグラフを作成します。グラフは「vert」の下に作成される「timeseries_plots」ディレクトリに出力されます。

## 4.3. plot_vert_lidar-wrf_timeseries.py
大橋を含むメッシュを対象に、Doppler LIDARの観測高度毎にDoppler LIDARの観測値とWRFの計算値の風向・風速の時系列データのグラフを作成します。WRFの計算値はDoppler LIDARの観測高度と最も近いものを選び、高度による補正は行っていません。グラフは「vert」の下に作成される「lidar-wrf_timeseries_plots」ディレクトリに出力されます。

## 4.4. plot_vert_profiles.py
気象要素の鉛直プロファイルのグラフを作成します。グラフは「vert」の下に作成される「profile_plots」ディレクトリに出力されます。なお、全てのグラフを出力するには1時間程度要します。