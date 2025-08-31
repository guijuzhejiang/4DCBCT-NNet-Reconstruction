# N-net 学习配置帮助文档

`config.py` 文件包含了用于 N-net 学习的所有配置参数。这些参数被 `train_dataaug.py` 脚本读取和使用，以控制训练过程、数据集设置、模型架构、日志记录和硬件使用。

## 配置结构

`config.py` 文件主要由以下几个 Python 字典组成，每个字典负责管理特定类别的配置：

- `TRAINING_CONFIG`: 学习过程的核心参数。
- `DATASET_CONFIG`: 数据集的路径、类型和划分。
- `MODEL_CONFIG`: 模型的基本架构参数。
- `LOGGING_CONFIG`: 学习过程中的日志记录设置。
- `SCHEDULER_CONFIG`: 学习率调度器的相关参数。
- `DEVICE_CONFIG`: 硬件设备（GPU/CUDA）的使用设置。

## 配置详情

### 1. `TRAINING_CONFIG` (学习参数)

此字典包含控制神经网络学习过程的关键参数。

| 参数名              | 类型     | 默认值      | 描述                                         |
| :------------------ | :------- | :---------- | :------------------------------------------- |
| `train_batch_size`  | `int`    | `48`        | 训练数据加载器的批处理大小。                 |
| `val_batch_size`    | `int`    | `48`        | 验证数据加载器的批处理大小。                 |
| `num_workers`       | `int`    | `1`         | 数据加载器用于数据加载的子进程数。           |
| `epochs`            | `int`    | `50`        | 模型的总训练轮次。                           |
| `model_save_dir`    | `str`    | `./trained_model` | 训练好的模型保存的相对路径。             |
| `weight_l1`         | `float`  | `0.1`       | L1 损失在总损失中的权重。                  |
| `weight_percep`     | `float`  | `0.2`       | 感知损失（Perceptual Loss）在总损失中的权重。 |
| `weight_ssim`       | `float`  | `0.3`       | SSIM 损失在总损失中的权重。                |
| `weight_mse`        | `float`  | `0.4`       | MSE 损失在总损失中的权重。                 |
| `seed`              | `int`    | `42`        | 用于所有随机操作的随机种子，以确保结果可复现。 |

**示例:**

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

### 2. `DATASET_CONFIG` (数据集设置)

此字典定义了数据集的路径、类型和如何划分训练、验证和测试集。

| 参数名                  | 类型          | 默认值                 | 描述                                                   |
| :---------------------- | :------------ | :--------------------- | :----------------------------------------------------- |
| `data_root`             | `str`         | `/home/zzg/data/Medical/4D_Lung_CBCT_Hitachi/dataset/` | 数据集根目录的绝对路径。                               |
| `fov_type`              | `str`         | `FovS_180`             | 视野类型，可选值包括 `"FovL"`, `"FovS_180"`, `"FovS_360"`。 |
| `train_dataset_indices` | `list[int]`   | `list(range(0, 40))`   | 用于训练的数据集索引列表。                             |
| `val_dataset_indices`   | `list[int]`   | `list(range(40, 45))`  | 用于验证的数据集索引列表。                             |
| `test_dataset_indices`  | `list[int]`   | `[40]`                 | 用于测试的数据集索引列表。                             |
| `image_size`            | `tuple[int, int]` | `(512, 512)`           | 输入和输出图像的尺寸 (高度, 宽度)。                  |
| `image_number`          | `int`         | `384`                  | 数据集中每个图像的总数量。                             |

**示例:**

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

### 3. `MODEL_CONFIG` (模型设置)

此字典包含了模型架构相关的基本参数。

| 参数名            | 类型    | 默认值 | 描述                 |
| :---------------- | :------ | :----- | :------------------- |
| `input_channels`  | `int`   | `1`    | 模型输入图像的通道数。 |
| `output_channels` | `int`   | `1`    | 模型输出图像的通道数。 |

**示例:**

```python
MODEL_CONFIG = {
    'input_channels': 1,
    'output_channels': 1,
}
```

### 4. `LOGGING_CONFIG` (日志记录设置)

此字典配置了学习过程中的日志记录行为，包括是否使用 Wandb 和 TensorBoard。

| 参数名                | 类型      | 默认值            | 描述                                       |
| :-------------------- | :-------- | :---------------- | :----------------------------------------- |
| `use_wandb`           | `bool`    | `True`            | 是否使用 Weights & Biases (Wandb) 进行日志记录。 |
| `use_tensorboard`     | `bool`    | `True`            | 是否使用 TensorBoard 进行日志记录。        |
| `wandb_project`       | `str`     | `nnet-medical-ct` | Wandb 项目的名称。                         |
| `wandb_entity`        | `str/None` | `None`            | Wandb 实体或用户名称（可选）。             |
| `tensorboard_log_dir` | `str`     | `./logs`          | TensorBoard 日志文件保存的相对路径。       |

**示例:**

```python
LOGGING_CONFIG = {
    'use_wandb': True,
    'use_tensorboard': True,
    'wandb_project': 'nnet-medical-ct',
    'wandb_entity': None,
    'tensorboard_log_dir': './logs',
}
```

### 5. `SCHEDULER_CONFIG` (学习率调度器)

此字典定义了学习率调度器的类型及其相关参数。

| 参数名             | 类型     | 默认值    | 描述                                                         |
| :----------------- | :------- | :-------- | :----------------------------------------------------------- |
| `type`             | `str`    | `CyclicLR` | 学习率调度器的类型。可选值包括：`StepLR`, `ReduceLROnPlateau`, `CosineAnnealingWarmRestarts`, `CosineAnnealingLR`, `CyclicLR`, `OneCycleLR`。 |
| `step_size`        | `int`    | `1`       | `StepLR` 调度器中学习率衰减的步长。                          |
| `gamma`            | `float`  | `0.90`    | `StepLR` 调度器中学习率衰减的乘数。                          |
| `lr`               | `float`  | `1e-5`    | 初始学习率。                                                 |
| `min_lr`           | `float`  | `1e-6`    | 学习率的最小值。                                             |
| `max_lr`           | `float`  | `1e-4`    | 学习率的最大值（用于 `CyclicLR`, `OneCycleLR`）。           |
| `plateau_factor`   | `float`  | `0.5`     | `ReduceLROnPlateau` 调度器中学习率衰减的因子。               |
| `plateau_patience` | `int`    | `5`       | `ReduceLROnPlateau` 调度器中模型性能停止改善后等待的 epoch 数。 |
| `T_0`              | `int`    | `4`       | `CosineAnnealingWarmRestarts` 调度器中第一次重启的 epoch 数。 |
| `T_mult`           | `int`    | `2`       | `CosineAnnealingWarmRestarts` 调度器中每次重启后周期乘数。 |
| `pct_start`        | `float`  | `0.3`     | `OneCycleLR` 调度器中学习率从 `min_lr` 增加到 `max_lr` 所占总步数的比例。 |
| `epoch_size_up`    | `int`    | `1`       | `CyclicLR` 调度器中从 `base_lr` 到 `max_lr` 的半个周期所需的 epoch 数。 |
| `mode`             | `str`    | `triangular` | `CyclicLR` 调度器的模式，可选 `triangular`, `triangular2`, `exp_range`。 |

**示例 (CyclicLR):**

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

### 6. `DEVICE_CONFIG` (硬件设置)

此字典配置了硬件设备的使用，例如是否使用 CUDA 和自动混合精度 (AMP)。

| 参数名       | 类型    | 默认值 | 描述                                       |
| :----------- | :------ | :----- | :----------------------------------------- |
| `use_cuda`   | `bool`  | `True` | 是否使用 CUDA (GPU) 进行训练。             |
| `cuda_device`| `int`   | `0`    | 如果使用 CUDA，指定要使用的 GPU 设备索引。 |
| `use_amp`    | `bool`  | `True` | 是否启用自动混合精度 (Automatic Mixed Precision, AMP) 以加速训练并减少内存使用。 |

**示例:**

```python
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0,
    'use_amp': True
}
```

## 如何修改配置

您可以直接编辑 `config.py` 文件来修改这些配置。在 `train_dataaug.py` 脚本运行时，它会自动导入并使用这些配置。

**重要提示:**

- 在修改 `data_root` 时，请确保路径是正确的绝对路径或相对于项目根目录的正确路径。
- 运行数据完整性检查脚本 `check_data.py`：
  ```bash
  python check_data.py
  ```
  此脚本将使用 `config.py` 中 `DATASET_CONFIG['data_root']` 定义的路径作为数据集根目录，并检查其目录结构和文件数量是否满足训练要求。强烈建议在开始训练前运行此脚本。
- 更改 `fov_type` 可能需要对应的数据集文件存在。
- 根据您的硬件资源调整 `train_batch_size` 和 `val_batch_size`，以避免内存不足错误。
- 调整损失权重 (`weight_l1`, `weight_percep`, `weight_ssim`, `weight_mse`) 以优化模型性能。
- 调整学习率调度器参数时，请参考 PyTorch 文档或相关教程以获得最佳实践。

通过修改这些配置，您可以灵活地控制 N-net 模型的训练行为和性能。