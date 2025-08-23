# 4DCBCT-NNet-Reconstruction

## 日本語

### プロジェクト概要
本プロジェクトは、4DコーンビームCT（CBCT）医用画像の再構成のために、独自のNNetディープラーニングモデルを実装しています。データ前処理、EDA解析、モデル学習、検証、テストまで一貫したパイプラインを提供します。構成はモジュール化され、設定・モデル・データ・ユーティリティが明確に分離されています。

### 主な特徴
- **カスタムNNetモデル**：4D CBCTのアーチファクト低減と画像強調のための深層学習ネットワーク。
- **柔軟なデータパイプライン**：生の`.img`医用画像、先行画像・多相データに対応。
- **包括的なEDA**：データセットの可視化と統計解析。
- **設定可能な学習フロー**：全ハイパーパラメータとパスは`config.py`で一元管理。
- **評価・テスト**：検証・テストスクリプトを含み、画像生成に対応。
- **ロギング**：TensorBoardおよびWeights & Biasesによる実験管理。

### ディレクトリ構成
- `config.py`：データ・モデル・学習・ロギングの設定。
- `model_Nnet.py`：NNetモデルの定義。
- `train.py`、`train_dataaug.py`：学習スクリプト（データ拡張あり/なし）。
- `train_dataset_Nnet.py`、`test_dataset_Nnet.py`：学習・テスト用データセットクラス。
- `EDA.py`：データ解析と可視化。
- `test.py`：モデル推論と画像生成。
- `metrics.py`：カスタム評価指標。
- `img_reader.py`：`.img`ファイル用カスタムリーダー。
- `utils.py`：ユーティリティ関数。
- `experiments/`、`eda_results/`、`prediction/`：出力・ログディレクトリ。

### 必要環境
- Python 3.8以上
- PyTorch、MONAI、NumPy、Matplotlib、torchvision、PIL、scikit-image、imageio、wandb、torchsummary、pytorch_msssim、psutil

### 使い方
1. `config.py`でパスとパラメータを設定。
2. EDA解析を実行：  
   ```bash
   python EDA.py
   ```
3. モデル学習：  
   ```bash
   python train.py
   ```
   またはデータ拡張あり：  
   ```bash
   python train_dataaug.py
   ```
4. `validation.py`と`test.py`で検証・テストを実施。