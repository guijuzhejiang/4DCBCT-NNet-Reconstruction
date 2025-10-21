"""
N-net学習設定ファイル
"""

# 学習パラメータ
TRAINING_CONFIG = {
    'train_batch_size': 32,
    'val_batch_size': 48,
    'num_workers': 4,
    'epochs': 100,
    'model_save_dir': './trained_model',
    'weight_l1': 0.3,
    'weight_percep': 0.0,   #(若 VGG 在医学图像上效果差，可移除)
    'weight_ssim': 0.4,
    'weight_mse': 0.0,
    'weight_grad': 0.3,
    'seed': 42,
    'early_stopping_patience': 3,
}

# データセット設定
DATASET_CONFIG = {
    'data_root': '/home/zzg/data/Medical/4D_Lung_CBCT_Hitachi/dataset/',
    'train_fov_type': 'FovL',            #"FovL", "FovS_180", "FovS_360"
    'test_fov_type': 'FovL',
    'train_dataset_indices': list(range(0, 40)),
    # 'train_dataset_indices': [0],
    'val_dataset_indices': list(range(40, 45)),
    # 'val_dataset_indices': [40],
    'test_dataset_indices': list(range(45, 50)),
    # 'test_dataset_indices': [40],
    'image_size': (512, 512),
    'image_number': 384,
    'ScaleIntensityRange_a_min': -160,
    'ScaleIntensityRange_a_max': 240,
    'ScaleIntensityRange_b_min': -1.0,
    'ScaleIntensityRange_b_miax': 1.0,
}

# モデル設定
MODEL_CONFIG = {
    'input_channels': 1,
    'output_channels': 1,
}

# ロギング設定
LOGGING_CONFIG = {
    'use_wandb': True,
    'use_tensorboard': True,
    'wandb_project': 'nnet-medical-ct',
    'wandb_entity': None,
}

# 学習率スケジューラ
SCHEDULER_CONFIG = {
    'type': 'ReduceLROnPlateau',
    'step_size': 1,     #default 5
    'gamma': 0.90,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'max_lr': 1e-4,
    # ReduceLROnPlateau パラメータ
    'plateau_factor': 0.2,
    'plateau_patience': 1,
    'ReduceLR_min_lr': 1e-7,
    # CosineAnnealingWarmRestarts パラメータ
    'T_0': 1,
    'T_mult': 2,
    # OneCycleLR パラメータ
    'pct_start': 0.3,
    # CyclicLR パラメータ,半サイクルエポック数
    'epoch_size_up': 1,
    'mode': 'triangular',
}

# ハードウェア設定
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0,
}