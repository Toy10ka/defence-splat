# Defence-Splat
 Defence-Splatは、MITライセンスで公開されている [3DGS実装](https://github.com/joeyan/gaussian_splatting) を拡張し、敵対的摂動（Adversarial Perturbations）に対する防御評価を行うためのROIベースの画像フィルタリングを追加した実装です。


## 主な特徴

* **ROIフィルタリング**

  * `roi_filter.divide_calc` は入力画像をブロックに分割し、各ブロックの平均総変動（IV\_AVG）を計算します。指定された閾値を超えるブロックにはガウシアンブラーを適用します。

* **データセットローダーの拡張**

  * `ColmapData` は元の画像、ROIでフィルタリングされた画像、クリーンな参照画像をロードします。
  * データセットロード時に、フィルター処理された画像についてのIV\_AVG統計情報が表示されます。

* **設定可能なパラメータ**

  * フィルタリングの閾値 (`IV_AVG_THRESHOLD`) と画像の分割数 (`DIVISIONS`) は、`SplatConfig` 内で調整可能です。

* **トレーニングの更新**

  * `SplatTrainer` は3種類の画像セットを正規化します。
  * トレーニング時にはフィルター処理された画像を用い、PSNR/SSIM評価はクリーンな画像を参照して行われます。


## ライセンス

本プロジェクトはMITライセンスに基づいて提供されています（詳細はLICENSEファイルを参照）。




