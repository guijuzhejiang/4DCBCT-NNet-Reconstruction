# 4D-CBCT運動アーチファクト除去システム

MONAIフレームワークを使用した4D-CBCTデータの運動アーチファクト除去のためのディープラーニングモデルです。

## 概要

このプロジェクトは、N-Netアーキテクチャを使用してCT画像から運動アーチファクトを除去するシステムです。
- **入力**: ノイズありCT画像 + 先行画像
- **出力**: クリーンなCT画像（アーチファクト除去済み）

## セットアップ

### 1. 環境構築

**Python バージョン**: 3.10

依存関係をインストールしてください：

```bash
pip install -r requirements.txt
```

### 2. データ設定

`config.py`ファイルを開き、`DATASET_CONFIG`の`data_root`パラメータをあなたのデータセットパスに設定してください：

```python
DATASET_CONFIG = {
    'data_root': '/your/dataset/path/',  # ← ここを変更
    ...
}
```

## 使用方法

### 学習実行

1. **設定**: `config.py`内の`DATASET_CONFIG`を学習用データセットに合わせて設定してください
   ```python
   DATASET_CONFIG = {
       'data_root': '/your/dataset/path/',              # データの根目录（文字列）
       'train_fov_type': 'FovL',                       # "FovL", "FovS_180", "FovS_360"のいずれか（文字列）
       'train_dataset_indices': list(range(0, 40)),    # 0-39のインデックス（リスト）
       'val_dataset_indices': list(range(40, 45)),     # 40-44のインデックス（リスト）
       ...
   }
   ```

2. **実行**:
   ```bash
   python 001_train.py
   ```

3. **出力**: 学習を開始すると、以下の場所に実験結果が保存されます：
   - `experiments/Nnet/FOV_type/timestamp/`

### 評価実行

学習済みモデルを使用した推論と定量評価を行います：

1. **設定**: 
   - `002_inference.py`内の`OUTPUT_FOLDER`を学習済みモデルが保存されているディレクトリに設定してください
   ```python
   OUTPUT_FOLDER = 'experiments/Nnet/FOV_type/timestamp/'  # 学習済みモデルのパス
   ```
   
   - `config.py`内の`DATASET_CONFIG`を評価用データセットに合わせて設定してください
   ```python
   DATASET_CONFIG = {
       'data_root': '/your/dataset/path/',              # データの根目录（文字列）
       'test_fov_type': 'FovL',                        # "FovL", "FovS_180", "FovS_360"のいずれか（文字列）
       'test_dataset_indices': list(range(45, 50)),    # 45-49のインデックス（リスト）
       ...
   }
   ```
2. **実行**:

```bash
python 002_inference.py
```

3. **出力**: 評価結果は`OUTPUT_FOLDER/inference_results/`以下に保存されます：
   - `comparison_images/`: 比較画像（Prior | Noisy | Restored | GT）
   - `comparison_raw_images/`: RAW形式の比較データ
   - `evaluation_results/`: 定量評価結果（CSV、ボックスプロット）

### 複数推論結果の横断評価

複数の推論結果を横断的に評価し、モデル間の性能比較を行います：

1. **設定**: `101_evaluate_multi_infer_AllSubjects.py`内の`INFER_FOLDERS`に比較対象の推論結果フォルダパスを設定してください
   ```python
   INFER_FOLDERS = {
       'モデル名1': '/path/to/inference_results/comparison_raw_images/',
       'モデル名2': '/path/to/inference_results/comparison_raw_images/',
       'モデル名3': '/path/to/inference_results/comparison_raw_images/',
   }
   ```
   
   **注意**: 実行前に、コード内の`/your_path/`を実際のローカルディレクトリパスに変更してください。

2. **実行**:
   ```bash
   python 101_evaluate_multi_infer_AllSubjects.py
   ```

3. **出力**: 評価結果は`/your_path/20250903_Hitachi_SampleCode/output/timestamp_0101_MultiSubject_MultiInferEval/`以下に保存されます：
   - `VisualEvaluation/`: 全モデルの目視比較画像（subject別）
   - `LineProfileEvaluation/`: ラインプロファイル解析結果（グラフ・CSV）
   - `QuantitativeEvaluation/`: 定量評価結果（バイオリンプロット、統計データ）

### 主要設定

`config.py`で以下の設定を調整できます：

- **学習パラメータ**: バッチサイズ、エポック数、損失関数の重み
- **データセット設定**: データパス、FOVタイプ、学習・検証用インデックス
- **ロギング設定**: WandB、TensorBoard設定

### データ形式

データセットは以下の構造である必要があります：

```
data_root/
├── FovL/
│   ├── subject_0000/
│   │   ├── phase_00/
│   │   │   ├── img/     # ノイズありCT画像(.imgファイル)
│   │   │   └── gt/     # クリーンCT画像(.imgファイル)
│   │   └── prior/         # 先行画像(.imgファイル)
│   └── ...
└── ...
```

## モデル詳細

- **アーキテクチャ**: U-Net風エンコーダ-デコーダ構造
- **活性化関数**: PReLU
- **損失関数**: MSE + L1 + SSIM + Perceptual Loss
- **評価指標**: RMSE, MAE, PSNR, SSIM, MS-SSIM, CORR2

## ログとモニタリング

- **TensorBoard**: `experiments/*/*/*/tensorboard/`にログ保存
- **WandB**: クラウドロギング対応
- **モデルチェックポイント**: 最良検証性能時に自動保存

## 注意事項

- GPU使用を推奨（CUDA対応）
- 大容量メモリが必要（512×512医用画像処理のため）
- 早期停止機能搭載（過学習防止）