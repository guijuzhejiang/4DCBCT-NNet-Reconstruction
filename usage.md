# N-net 学習設定ヘルプドキュメント

`config.py` ファイルには、N-net 学習に必要なすべての設定パラメータが含まれています。これらのパラメータは `train_dataaug.py` スクリプトによって読み込まれ、学習プロセス、データセット設定、モデルアーキテクチャ、ロギング、ハードウェア使用を制御するために使用されます。

## 設定構造

`config.py` ファイルは、主に以下の Python 辞書で構成されており、それぞれが特定カテゴリの設定を管理しています。

- `TRAINING_CONFIG`: 学習プロセスのコアパラメータ。
- `DATASET_CONFIG`: データセットのパス、タイプ、および分割。
- `MODEL_CONFIG`: モデルの基本アーキテクチャパラメータ。
- `LOGGING_CONFIG`: 学習中のロギング設定。
- `SCHEDULER_CONFIG`: 学習率スケジューラに関連するパラメータ。
- `DEVICE_CONFIG`: ハードウェアデバイス（GPU/CUDA）の使用設定。

## 設定詳細

### 1. `TRAINING_CONFIG` (学習パラメータ)

この辞書には、ニューラルネットワークの学習プロセスを制御するための主要なパラメータが含まれています。

| パラメータ名              | 型     | デフォルト値 | 説明                                       |
| :------------------ | :------- | :----------- | :----------------------------------------- |
| `train_batch_size`  | `int`    | `48`         | 訓練データローダーのバッチサイズ。         |
| `val_batch_size`    | `int`    | `48`         | 検証データローダーのバッチサイズ。         |
| `num_workers`       | `int`    | `1`          | データロードに使用されるデータローダーのサブプロセス数。 |
| `epochs`            | `int`    | `50`         | モデルの総学習エポック数。                 |
| `model_save_dir`    | `str`    | `./trained_model` | 学習済みモデルが保存される相対パス。       |
| `weight_l1`         | `float`  | `0.1`        | 総損失における L1 損失の重み。             |
| `weight_percep`     | `float`  | `0.2`        | 総損失における知覚損失（Perceptual Loss）の重み。 |
| `weight_ssim`       | `float`  | `0.3`        | 総損失における SSIM 損失の重み。           |
| `weight_mse`        | `float`  | `0.4`        | 総損失における MSE 損失の重み。            |
| `seed`              | `int`    | `42`         | 結果の再現性を保証するためのすべてのランダム操作に使用されるシード。 |

**例:**

```python
TRAINING_CONFIG = {
    'train_batch_size': 48,
    'val_batch_size': 48,
    'num_workers': 1,
    'epochs': 50,
    'model_save_dir': './trained_model',
    'weight_l1': 0.1,
    'weight_percep': 0.2,
    'weight_ssim': 0.3,
    'weight_mse': 0.4,
    'seed': 42,
}
```

### 2. `DATASET_CONFIG` (データセット設定)

この辞書は、データセットのパス、タイプ、および訓練、検証、テストセットの分割方法を定義します。

| パラメータ名                  | 型          | デフォルト値                   | 説明                                                           |
| :---------------------- | :------------ | :----------------------------- | :------------------------------------------------------------- |
| `data_root`             | `str`         | `/home/zzg/data/Medical/4D_Lung_CBCT_Hitachi/dataset/` | データセットのルートディレクトリの絶対パス。                   |
| `fov_type`              | `str`         | `FovS_180`                     | 視野タイプ。選択肢は `"FovL"`, `"FovS_180"`, `"FovS_360"`。     |
| `train_dataset_indices` | `list[int]`   | `list(range(0, 40))`           | 訓練に使用されるデータセットのインデックスリスト。             |
| `val_dataset_indices`   | `list[int]`   | `list(range(40, 45))`          | 検証に使用されるデータセットのインデックスリスト。             |
| `test_dataset_indices`  | `list[int]`   | `[40]`                         | テストに使用されるデータセットのインデックスリスト。           |
| `image_size`            | `tuple[int, int]` | `(512, 512)`                   | 入力および出力画像のサイズ (高さ, 幅)。                        |
| `image_number`          | `int`         | `384`                          | データセット内の各画像の総数。                                 |

**例:**

```python
DATASET_CONFIG = {
    'data_root': '/home/zzg/data/Medical/4D_Lung_CBCT_Hitachi/dataset/',
    'fov_type': 'FovS_180',
    'train_dataset_indices': list(range(0, 40)),
    'val_dataset_indices': list(range(40, 45)),
    'test_dataset_indices': [40],
    'image_size': (512, 512),
    'image_number': 384,
}
```

### 3. `MODEL_CONFIG` (モデル設定)

この辞書には、モデルアーキテクチャに関連する基本的なパラメータが含まれています。

| パラメータ名            | 型    | デフォルト値 | 説明                 |
| :---------------- | :------ | :----- | :------------------- |
| `input_channels`  | `int`   | `1`    | モデル入力画像のチャンネル数。 |
| `output_channels` | `int`   | `1`    | モデル出力画像のチャンネル数。 |

**例:**

```python
MODEL_CONFIG = {
    'input_channels': 1,
    'output_channels': 1,
}
```

### 4. `LOGGING_CONFIG` (ロギング設定)

この辞書は、Wandb および TensorBoard の使用を含む、学習中のロギング動作を設定します。

| パラメータ名                | 型      | デフォルト値          | 説明                                            |
| :-------------------- | :-------- | :-------------------- | :---------------------------------------------- |
| `use_wandb`           | `bool`    | `True`                | Weights & Biases (Wandb) を使用してロギングするかどうか。 |
| `use_tensorboard`     | `bool`    | `True`                | TensorBoard を使用してロギングするかどうか。    |
| `wandb_project`       | `str`     | `nnet-medical-ct`     | Wandb プロジェクト名。                          |
| `wandb_entity`        | `str/None` | `None`                | Wandb エンティティまたはユーザー名（オプション）。 |

**例:**

```python
LOGGING_CONFIG = {
    'use_wandb': True,
    'use_tensorboard': True,
    'wandb_project': 'nnet-medical-ct',
    'wandb_entity': None,
}
```

### 5. `SCHEDULER_CONFIG` (学習率スケジューラ)

この辞書は、学習率スケジューラのタイプと関連パラメータを定義します。

| パラメータ名             | 型     | デフォルト値    | 説明                                                           |
| :----------------- | :------- | :-------------- | :------------------------------------------------------------- |
| `type`             | `str`    | `CyclicLR`      | 学習率スケジューラのタイプ。選択肢は：`StepLR`, `ReduceLROnPlateau`, `CosineAnnealingWarmRestarts`, `CosineAnnealingLR`, `CyclicLR`, `OneCycleLR`。 |
| `step_size`        | `int`    | `1`             | `StepLR` スケジューラにおける学習率減衰のステップサイズ。    |
| `gamma`            | `float`  | `0.90`          | `StepLR` スケジューラにおける学習率減衰の乗数。              |
| `lr`               | `float`  | `1e-5`          | 初期学習率。                                                   |
| `min_lr`           | `float`  | `1e-6`          | 学習率の最小値。                                               |
| `max_lr`           | `float`  | `1e-4`          | 学習率の最大値（`CyclicLR`, `OneCycleLR` で使用）。           |
| `plateau_factor`   | `float`  | `0.5`           | `ReduceLROnPlateau` スケジューラにおける学習率減衰の因子。   |
| `plateau_patience` | `int`    | `5`             | `ReduceLROnPlateau` スケジューラにおいて、モデル性能が改善を停止した後、待機するエポック数。 |
| `T_0`              | `int`    | `4`             | `CosineAnnealingWarmRestarts` スケジューラにおける初回再起動までのエポック数。 |
| `T_mult`           | `int`    | `2`             | `CosineAnnealingWarmRestarts` スケジューラにおける各再起動後の周期乗数。 |
| `pct_start`        | `float`  | `0.3`           | `OneCycleLR` スケジューラにおいて、学習率が `min_lr` から `max_lr` に増加する総ステップの割合。 |
| `epoch_size_up`    | `int`    | `1`             | `CyclicLR` スケジューラにおいて、`base_lr` から `max_lr` までの半周期に必要なエポック数。 |
| `mode`             | `str`    | `triangular`    | `CyclicLR` スケジューラのモード。選択肢は `triangular`, `triangular2`, `exp_range`。 |

**例 (CyclicLR):**

```python
SCHEDULER_CONFIG = {
    'type': 'CyclicLR',
    'step_size': 1,
    'gamma': 0.90,
    'lr': 1e-5,
    'min_lr': 1e-6,
    'max_lr': 1e-4,
    'plateau_factor': 0.5,
    'plateau_patience': 5,
    'T_0': 4,
    'T_mult': 2,
    'pct_start': 0.3,
    'epoch_size_up': 1,
    'mode': 'triangular',
}
```

### 6. `DEVICE_CONFIG` (ハードウェア設定)

この辞書は、CUDA や自動混合精度 (AMP) の使用など、ハードウェアデバイスの使用を設定します。

| パラメータ名       | 型    | デフォルト値 | 説明                                       |
| :----------- | :------ | :----- | :----------------------------------------- |
| `use_cuda`   | `bool`  | `True` | CUDA (GPU) を使用して学習するかどうか。    |
| `cuda_device`| `int`   | `0`    | CUDA を使用する場合、使用する GPU デバイスのインデックス。 |
| `use_amp`    | `bool`  | `True` | 自動混合精度 (Automatic Mixed Precision, AMP) を有効にして、学習を高速化し、メモリ使用量を削減するかどうか。 |

**例:**

```python
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0,
    'use_amp': True
}
```

## 設定の変更方法

`config.py` ファイルを直接編集することで、これらの設定を変更できます。`train_dataaug.py` スクリプトが実行されると、これらの設定が自動的にインポートされ、使用されます。

**重要:**

- `data_root` を変更する際は、パスが正しい絶対パスであるか、またはプロジェクトのルートディレクトリに対する正しい相対パスであることを確認してください。
- データセット整合性チェックスクリプト `check_data.py` を実行してください：
  ```bash
  python check_data.py
  ```
  このスクリプトは、`config.py` の `DATASET_CONFIG['data_root']` で定義されたパスをデータセットのルートディレクトリとして使用し、そのディレクトリ構造とファイル数が学習要件を満たしているか検証します。トレーニングを開始する前にこのスクリプトを実行することを強く推奨します。
- `fov_type` を変更すると、対応するデータセットファイルが必要になる場合があります。
- メモリ不足エラーを避けるために、ハードウェアリソースに合わせて `train_batch_size` および `val_batch_size` を調整してください。
- モデルの性能を最適化するために、損失の重み (`weight_l1`, `weight_percep`, `weight_ssim`, `weight_mse`) を調整してください。
- 学習率スケジューラのパラメータを調整する際は、PyTorch のドキュメントまたは関連するチュートリアルを参照してベストプラクティスを確認してください。

これらの設定を変更することで、N-net モデルの学習動作と性能を柔軟に制御できます。