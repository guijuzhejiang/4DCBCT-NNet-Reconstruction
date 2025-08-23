# 4DCBCT-NNet-Reconstruction

## English

### Project Overview
This project implements a deep learning pipeline for 4D Cone-Beam CT (CBCT) medical image reconstruction using a custom NNet architecture. The workflow includes data preprocessing, exploratory data analysis (EDA), model training, validation, and testing. The codebase is modular, with clear separation of configuration, model definition, data handling, and utilities.

### Main Features
- **Custom NNet Model**: Deep neural network for artifact reduction and image enhancement in 4D CBCT.
- **Flexible Data Pipeline**: Handles raw `.img` medical images, supports prior and phase data.
- **Comprehensive EDA**: Visualization and statistical analysis of dataset characteristics.
- **Configurable Training**: All hyperparameters and paths are managed via `config.py`.
- **Evaluation & Testing**: Includes scripts for validation and test image generation.
- **Logging**: Supports TensorBoard and Weights & Biases for experiment tracking.

### Directory Structure
- `config.py`: Configuration for dataset, model, training, and logging.
- `model_Nnet.py`: NNet model architecture.
- `train.py`, `train_dataaug.py`: Training scripts (with/without data augmentation).
- `train_dataset_Nnet.py`, `test_dataset_Nnet.py`: Dataset classes for training and testing.
- `EDA.py`: Exploratory data analysis and visualization.
- `test.py`: Model inference and image generation.
- `metrics.py`: Custom evaluation metrics.
- `img_reader.py`: Custom image reader for `.img` files.
- `utils.py`: Utility functions.
- `experiments/`, `eda_results/`, `prediction/`: Output directories for results and logs.

### Requirements
- Python 3.8+
- PyTorch, MONAI, NumPy, Matplotlib, torchvision, PIL, scikit-image, imageio, wandb, torchsummary, pytorch_msssim, psutil

### Usage
1. Configure paths and parameters in `config.py`.
2. Run EDA:  
   ```bash
   python EDA.py
   ```
3. Train the model:  
   ```bash
   python train.py
   ```
   or with data augmentation:  
   ```bash
   python train_dataaug.py
   ```
4. Validate and test using `validation.py` and `test.py`.