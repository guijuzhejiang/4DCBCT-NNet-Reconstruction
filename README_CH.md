# 4DCBCT-NNet-Reconstruction

## 中文

### 项目简介
本项目基于4D锥形束CT（CBCT）医学影像，构建了NNet深度学习网络，实现伪影去除与图像增强。项目包含数据预处理、EDA分析、模型训练、验证与测试，结构清晰，模块分明。

### 主要特性
- **自定义NNet模型**：针对4D CBCT伪影去除与图像增强的深度神经网络。
- **灵活数据管道**：支持原始`.img`医学图像，兼容先验与多相位数据。
- **数据集完整性检查**：验证数据集结构和文件数量，确保满足训练要求。
- **全面EDA分析**：数据集可视化与统计分析。
- **可配置训练流程**：所有超参数与路径集中于`config.py`管理。
- **评估与测试**：包含验证与测试脚本，支持图像生成。
- **日志记录**：支持TensorBoard与Weights & Biases实验追踪。

### 目录结构
- `config.py`：数据、模型、训练、日志配置。
- `model_Nnet.py`：NNet模型结构定义。
- `check_data.py`：数据集完整性检查脚本。
- `train.py`、`train_dataaug.py`：训练脚本（含/不含数据增强）。
- `train_dataset_Nnet.py`、`test_dataset_Nnet.py`：训练与测试数据集类。
- `EDA.py`：数据分析与可视化。
- `test.py`：模型推理与图像生成。
- `metrics.py`：自定义评估指标。
- `img_reader.py`：自定义`.img`文件读取器。
- `utils.py`：工具函数。
- `experiments/`、`eda_results/`、`prediction/`：结果与日志输出目录。

### 环境依赖
- Python 3.8+
- PyTorch、MONAI、NumPy、Matplotlib、torchvision、PIL、scikit-image、imageio、wandb、torchsummary、pytorch_msssim、psutil

### 使用方法
1. 在`config.py`中配置路径与参数。
2. 运行数据集完整性检查：
   ```bash
   python check_data.py
   ```
   此脚本会使用`config.py`中的`DATASET_CONFIG['data_root']`作为根路径，检查数据集的目录结构和文件数量是否满足训练要求。
3. 运行EDA分析：  
   ```bash
   python EDA.py
   ```
4. 训练模型：  
   ```bash
   python train.py
   ```
   或使用数据增强：  
   ```bash
   python train_dataaug.py
   ```
5. 使用`validation.py`和`test.py`进行验证与测试。