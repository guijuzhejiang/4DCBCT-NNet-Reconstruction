"""
Configuration file for N-net training
"""

# Training parameters
TRAINING_CONFIG = {
    'train_batch_size': 48, #48
    'val_batch_size': 48,
    'num_workers': 8,
    'epochs': 50,
    'model_save_dir': './trained_model',
    'weight_l1': 0.1,
    'weight_percep': 0.2,
    'weight_ssim': 0.3,
    'weight_mse': 0.4,
    'seed': 42,
}

# Dataset configuration
DATASET_CONFIG = {
    'data_root': '/media/zzg/GJ_disk01/data/Medical/4D_Lung_CBCT_Hitachi/dataset',
    'LMDB_cache_dir_train': '/media/zzg/GJ_disk02/data/Medical/4D_Lung_CBCT_Hitachi/LMDB_cache_dir_train',
    'LMDB_cache_dir_val': '/media/zzg/GJ_disk02/data/Medical/4D_Lung_CBCT_Hitachi/LMDB_cache_dir_val',
    'map_size_train': 500,        # 500GB空间预留
    'map_size_val': 100,          # 100GB空间预留
    'train_dataset_indices': list(range(0, 40)),
    # 'train_dataset_indices': [0],
    'val_dataset_indices': list(range(40, 45)),
    # 'val_dataset_indices': [40],
    # 'test_dataset_indices': list(range(45, 50)),
    'test_dataset_indices': [40],
    'image_size': (512, 512),
    'image_number': 384,
}

# Model configuration
MODEL_CONFIG = {
    'input_channels': 1,
    'output_channels': 1,
}

# Logging configuration
LOGGING_CONFIG = {
    'use_wandb': True,
    'use_tensorboard': True,
    'wandb_project': 'nnet-medical-ct',
    'wandb_entity': None,  # Set your wandb entity/username if needed
    'tensorboard_log_dir': './logs',
}

# Learning rate scheduler
SCHEDULER_CONFIG = {
    'type': 'StepLR',
    'step_size': 1,     #default 5
    'gamma': 0.90,
    'lr': 1e-5,
    'min_lr': 1e-6,
    'max_lr': 1e-4,
    # ReduceLROnPlateau 参数
    'plateau_factor': 0.5,
    'plateau_patience': 5,
    # CosineAnnealingWarmRestarts 参数
    'T_0': 4,
    'T_mult': 2,
    # OneCycleLR 参数
    'pct_start': 0.3,
    # CyclicLR 参数,代表半循环的epoch数
    'epoch_size_up': 0.5,
    'mode': 'triangular',
}

# Hardware configuration
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0,
    'use_amp': True  # 新增自动混合精度选项
}