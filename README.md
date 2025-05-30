# Defence-Splat

[3D Gaussian Splatting for Real‑Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) を拡張した実装です。

このリポジトリはMITライセンスで公開されている [元の実装](https://github.com/joeyan/gaussian_splatting) を基に、敵対的摂動（Adversarial Perturbations）に対する防御評価を行うためのROIベースの画像フィルタリングを追加しています。

## 主な特徴

* **ROIフィルタリング**

  * `roi_filter.divide_calc` は入力画像をブロックに分割し、各ブロックの平均総変動（IV\_AVG）を計算します。指定された閾値を超えるブロックにはガウシアンブラーを適用します。

* **データセットローダーの拡張**

  * `ColmapData` は元の画像、ROIでフィルタリングされた画像、クリーンな参照画像をロードします。
  * フィルター処理された画像は `processed_poison` ディレクトリに保存され、データセットロード時にIV\_AVG統計情報が表示されます。

* **設定可能なパラメータ**

  * フィルタリングの閾値 (`IV_AVG_THRESHOLD`) と画像の分割数 (`DIVISIONS`) は、`SplatConfig` 内で調整可能です。

* **トレーニングの更新**

  * `SplatTrainer` は3種類の画像セットを正規化します。
  * トレーニング時にはフィルター処理された画像を用い、PSNR/SSIM評価はクリーンな画像を参照して行われます。


## 使用方法

COLMAP形式のデータセットを用意します。

* フル解像度の画像を`images`フォルダに、ダウンサンプリングした画像を`images_<N>`フォルダに、`sparse/0`フォルダには `cameras.bin`、`images.bin`、`points3D.bin` を配置します。
* クリーンな参照画像は `images_<N>clean` に配置します。

トレーニングの実行例（7,000回のイテレーションを使用）：

```bash
python colmap_splat.py 7k --dataset_path <dataset_dir> --downsample_factor 4
```

出力された指標とフィルタリングされた画像は `splat_output/` に保存されます。最終的なGaussianデータは `gaussians_final.pt` として保存され、最高テストPSNRが報告されます。

## ライセンス

本プロジェクトはMITライセンスに基づいて提供されています（詳細はLICENSEファイルを参照）。




