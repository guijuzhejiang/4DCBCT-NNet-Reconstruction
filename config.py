"""
N-net学習設定ファイル
"""

# 学習パラメータ
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

# データセット設定
DATASET_CONFIG = {
    # 'data_root': '/media/zzg/GJ_disk01/data/Medical/4D_Lung_CBCT_Hitachi/dataset',
    'data_root': '/home/zzg/data/Medical/4D_Lung_CBCT_Hitachi/dataset/',
    'LMDB_cache_dir_train': '/media/zzg/GJ_disk02/data/Medical/4D_Lung_CBCT_Hitachi/LMDB_cache_dir_train',
    'LMDB_cache_dir_val': '/media/zzg/GJ_disk02/data/Medical/4D_Lung_CBCT_Hitachi/LMDB_cache_dir_val',
    'map_size_train': 1000,        # 500GB空間を確保
    'map_size_val': 100,          # 100GB空間を確保
    'fov_type': 'FovS_180',            #"FovL", "FovS_180", "FovS_360"
    'train_dataset_indices': list(range(0, 40)),
    # 'train_dataset_indices': [0],
    'val_dataset_indices': list(range(40, 45)),
    # 'val_dataset_indices': [40],
    # 'test_dataset_indices': list(range(45, 50)),
    'test_dataset_indices': [40],
    'image_size': (512, 512),
    'image_number': 384,
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
    'wandb_entity': None,  # 必要に応じてwandbエンティティ/ユーザー名を設定
    'tensorboard_log_dir': './logs',
}

# 学習率スケジューラ
SCHEDULER_CONFIG = {
    'type': 'CyclicLR',
    'step_size': 1,     #default 5
    'gamma': 0.90,
    'lr': 1e-5,
    'min_lr': 1e-6,
    'max_lr': 1e-4,
    # ReduceLROnPlateau パラメータ
    'plateau_factor': 0.5,
    'plateau_patience': 5,
    # CosineAnnealingWarmRestarts パラメータ
    'T_0': 4,
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
    'use_amp': True  # 自動混合精度オプションを追加
}