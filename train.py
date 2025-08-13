"""
N-net Training Script for Medical CT Data
Refactored version with improved logging, configuration management, and code organization.
"""

import os
import shutil
import numpy as np
import torch
from metrics import Corr2Metric
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision
import random
import warnings

warnings.filterwarnings("ignore")

# Import custom modules
from model_Nnet import Nnet
from TrainDataset_Nnet import Nnet_Dataset, LoadRawImgSlice
from config import TRAINING_CONFIG, DATASET_CONFIG, MODEL_CONFIG, LOGGING_CONFIG, SCHEDULER_CONFIG, DEVICE_CONFIG
from utils import setup_device, save_model
from monai.losses import SSIMLoss, PerceptualLoss, MultiScaleLoss
from torch.nn import MSELoss, L1Loss
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord
)
from monai.data import CacheDataset, ThreadDataLoader
from monai.metrics import SSIMMetric, MAEMetric, PSNRMetric, RMSEMetric, MultiScaleSSIMMetric
from datetime import datetime
import wandb
from img_reader import CustomIMGReader


class NnetTrainer:
    """N-net trainer class with integrated logging and monitoring."""

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.device = setup_device(DEVICE_CONFIG['use_cuda'], DEVICE_CONFIG['cuda_device'])
        # 初始化指标计算器
        self.ssim_metric = SSIMMetric(spatial_dims=2, reduction="mean")
        self.msssim_metric = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, reduction="mean") #因为图像归一化到 [0,1]
        self.mae_metric = MAEMetric(reduction="mean")
        self.psnr_metric = PSNRMetric(max_val=1.0, reduction="mean")
        self.rmse_metric = RMSEMetric(reduction="mean")
        self.corr2_metric = Corr2Metric(reduction="mean")
        self.setup_logging()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        # 变量跟踪最佳验证性能
        self.best_val_loss = float('inf')
        self.best_val_metrics = None
        self.best_epoch = 0

    def setup_logging(self):
        """Setup wandb and tensorboard logging."""
        # Tensorboard setup
        if LOGGING_CONFIG['use_tensorboard']:
            # 使用实验目录下的子目录
            tb_log_dir = os.path.join(self.experiment_dir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_log_dir)
            print(f"Tensorboard logs will be saved to: {tb_log_dir}")
        else:
            self.tb_writer = None

        # Wandb setup
        if LOGGING_CONFIG['use_wandb']:
            # 使用实验目录名作为wandb的run name
            run_name = os.path.basename(self.experiment_dir)
            wandb.init(
                project=LOGGING_CONFIG['wandb_project'],
                entity=LOGGING_CONFIG['wandb_entity'],
                config={
                    **TRAINING_CONFIG,
                    **DATASET_CONFIG,
                    **MODEL_CONFIG,
                    **SCHEDULER_CONFIG
                },
                name=run_name,  # 设置run name
                dir=self.experiment_dir  # 设置wandb日志保存到实验目录
            )
            self.use_wandb = True
            print("Wandb logging initialized")
        else:
            self.use_wandb = False
            if LOGGING_CONFIG['use_wandb']:
                print("Wandb requested but not available")

    # 在setup_data方法中
    def setup_data(self):
        # 创建reader
        reader = CustomIMGReader(image_shape=(1, 512, 512), dtype=np.int16, output_dtype=float)
        # 训练集transforms - 针对伪影任务的增强
        train_val_transforms = Compose([
            LoadImaged(keys=["img", "prior", "label"], reader=reader),
            EnsureChannelFirstd(keys=["img", "prior", "label"]),
            ScaleIntensityRanged(
                keys=["img", "prior", "label"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # 转换为Tensor
            ToTensord(keys=["img", "prior", "label"])
        ])


        """Setup data loaders."""
        print("Setting up datasets...")

        train_indices = DATASET_CONFIG['train_dataset_indices']
        val_indices = DATASET_CONFIG['val_dataset_indices']

        print("train_indices:", train_indices)
        train_dataset = Nnet_Dataset(DATASET_CONFIG['data_root'], train_indices)

        print("val_indices:", val_indices)
        val_dataset = Nnet_Dataset(DATASET_CONFIG['data_root'], val_indices)

        print(f"train Dataset sizes:{len(train_dataset)}, validation Dataset sizes:{len(val_dataset)}")

        # 分割数据集并应用transforms
        self.train_dataset = CacheDataset(
            data=train_dataset,
            transform=train_val_transforms,
            cache_rate=0.1,  # 将所有数据预处理后缓存到内存，训练时直接读取内存，速度最快，但内存占用高。
            num_workers=8,
            progress=True
        )
        self.val_dataset = CacheDataset(
            data=val_dataset,
            transform=train_val_transforms,
            cache_rate=0.2,
            num_workers=4,
            progress=True
        )

        # MONAI优化后的DataLoader
        self.train_loader = ThreadDataLoader(
            self.train_dataset,
            batch_size=TRAINING_CONFIG['train_batch_size'],
            shuffle=True,
            num_workers=TRAINING_CONFIG['num_workers'],
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
            persistent_workers=True
        )

        self.val_loader = ThreadDataLoader(
            self.val_dataset,
            batch_size=TRAINING_CONFIG['val_batch_size'],
            num_workers=TRAINING_CONFIG['num_workers'],
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2
        )

    def setup_model(self):
        """Setup model and move to device."""
        print("Setting up model...")

        self.model = Nnet()
        self.model = self.model.to(self.device)

        # Print model summary
        print('Model architecture:' + "=" * 60)
        summary(self.model, [(MODEL_CONFIG['input_channels'], *DATASET_CONFIG['image_size']),
                             (MODEL_CONFIG['input_channels'], *DATASET_CONFIG['image_size'])])
        print("=" * 60)

        # Log model to tensorboard
        if self.tb_writer:
            dummy_input1 = torch.randn(1, MODEL_CONFIG['input_channels'], *DATASET_CONFIG['image_size']).to(self.device)
            dummy_input2 = torch.randn(1, MODEL_CONFIG['input_channels'], *DATASET_CONFIG['image_size']).to(self.device)
            self.tb_writer.add_graph(self.model, (dummy_input1, dummy_input2))

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def setup_optimizer(self):
        """Setup optimizer and loss function."""
        batches_per_epoch = len(self.train_loader)
        total_batches = TRAINING_CONFIG['epochs'] * batches_per_epoch
        weight_mse = TRAINING_CONFIG['weight_mse']
        weight_l1 = TRAINING_CONFIG['weight_l1']
        weight_percep = TRAINING_CONFIG['weight_percep']
        weight_ssim = TRAINING_CONFIG['weight_ssim']
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=SCHEDULER_CONFIG['lr']
        )

        # 根据配置选择学习率调度器
        scheduler_type = SCHEDULER_CONFIG.get('type', 'StepLR')
        print(f"Using {scheduler_type} learning rate scheduler")

        if scheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=SCHEDULER_CONFIG['step_size'] * batches_per_epoch,
                gamma=SCHEDULER_CONFIG['gamma']
            )
        # elif scheduler_type == 'ReduceLROnPlateau':         #停止改善时降低学习率
        #     # ReduceLROnPlateau通常在每个epoch后更新，但我们在validate_epoch后更新
        #     self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #         self.optimizer,
        #         mode='min',
        #         factor=SCHEDULER_CONFIG['plateau_factor'],
        #         patience=SCHEDULER_CONFIG['plateau_patience'],
        #         verbose=True,
        #         min_lr=SCHEDULER_CONFIG['min_lr']
        #     )
        elif scheduler_type == 'CosineAnnealingWarmRestarts':           #热重启的余弦退火
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=SCHEDULER_CONFIG['T_0'] * batches_per_epoch,
                T_mult=SCHEDULER_CONFIG['T_mult'],
                eta_min=SCHEDULER_CONFIG['min_lr']
            )
        elif scheduler_type == 'CosineAnnealingLR':         #余弦退火
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_batches,
                eta_min=SCHEDULER_CONFIG['min_lr']
            )
        elif scheduler_type == 'CyclicLR':          # 循环学习率
            self.scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=SCHEDULER_CONFIG['min_lr'],  # 基础学习率
                max_lr=SCHEDULER_CONFIG['max_lr'],  # 最大学习率
                step_size_up=SCHEDULER_CONFIG['epoch_size_up'] * batches_per_epoch,  # 从基础到最大所需的步数
                mode=SCHEDULER_CONFIG['mode']  # 三角模式
            )
        elif scheduler_type == 'OneCycleLR':          # 循环学习率
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=SCHEDULER_CONFIG['max_lr'],
                total_steps=total_batches,
                pct_start=SCHEDULER_CONFIG['pct_start'],
                anneal_strategy='cos',
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=SCHEDULER_CONFIG['T_0'] * batches_per_epoch,        # 初始重启周期
                T_mult=SCHEDULER_CONFIG['T_mult'],  # 周期倍增因子
                eta_min=SCHEDULER_CONFIG['min_lr']
            )

        self.mse_loss = MSELoss().to(self.device)
        self.L1_loss = L1Loss().to(self.device)
        # 添加MONAI医学图像专用损失函数
        self.ssim_loss = SSIMLoss(spatial_dims=2).to(self.device)
        # 伪影去除专用损失
        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex").to(self.device)

        # 组合损失函数
        def combined_loss(pred, target):
            mse = self.mse_loss(pred, target)
            l1 = self.L1_loss(pred, target)
            ssim = self.ssim_loss(pred, target)
            percep = self.perceptual_loss(pred, target)
            loss_total = weight_mse * mse + weight_l1 * l1 + weight_percep * percep + weight_ssim * ssim
            return loss_total

        self.criterion = combined_loss

    def calculate_metrics(self, pred, target):
        # 重置指标
        self.ssim_metric.reset()
        self.msssim_metric.reset()
        self.mae_metric.reset()
        self.psnr_metric.reset()
        self.rmse_metric.reset()
        self.corr2_metric.reset()

        pred = pred.float()
        target = target.float()

        # 批量计算指标
        self.ssim_metric(pred, target)
        ssim_val = self.ssim_metric.aggregate().item()

        self.msssim_metric(pred, target)
        msssim_val = self.msssim_metric.aggregate().item()

        self.mae_metric(pred, target)
        mae_val = self.mae_metric.aggregate().item()

        self.psnr_metric(pred, target)
        psnr_val = self.psnr_metric.aggregate().item()

        self.rmse_metric(pred, target)
        rmse_val = self.rmse_metric.aggregate().item()

        self.corr2_metric(pred, target)
        corr2_val = self.corr2_metric.aggregate().item()

        return {
            "rmse": rmse_val,
            "mae": mae_val,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "msssim": msssim_val,
            "corr2": corr2_val
        }

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        print(f"\nEpoch {epoch + 1}/{TRAINING_CONFIG['epochs']} Training")

        for i, batch in enumerate(self.train_loader):
            # batch是字典形式
            images = batch["img"].to(self.device)
            prior = batch["prior"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            try:
                prediction = self.model(images, prior)
                loss = self.criterion(prediction, labels)
                # 反向传播
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # 优化器更新
                self.optimizer.step()
                # 学习率调度器更新
                self.scheduler.step()
                # 损失
                batch_loss = loss.item()
                # Calculate metrics
                batch_metrics = self.calculate_metrics(prediction, labels)

                print(f'[Epoch {epoch + 1} Batch {i + 1}] LR: {self.optimizer.param_groups[0]["lr"]}, '
                      f'Loss: {batch_loss:.6f}, RMSE: {batch_metrics["rmse"]:.6f}, MAE: {batch_metrics["mae"]:.6f}, '
                      f'PSNR: {batch_metrics["psnr"]:.6f}, SSIM: {batch_metrics["ssim"]:.6f}, MSSSIM: {batch_metrics["msssim"]:.6f}, CORR2: {batch_metrics["corr2"]:.6f}')
                # Log to tensorboard
                if self.tb_writer:
                    global_step = epoch * len(self.train_loader) + i
                    self.tb_writer.add_scalar('Train/Loss', batch_loss, global_step)
                    self.tb_writer.add_scalar('Train/RMSE', batch_metrics['rmse'], global_step)
                    self.tb_writer.add_scalar('Train/MAE', batch_metrics['mae'], global_step)
                    self.tb_writer.add_scalar('Train/PSNR', batch_metrics['psnr'], global_step)
                    self.tb_writer.add_scalar('Train/SSIM', batch_metrics['ssim'], global_step)
                    self.tb_writer.add_scalar('Train/MSSSIM', batch_metrics['msssim'], global_step)
                    self.tb_writer.add_scalar('Train/CORR2', batch_metrics['corr2'], global_step)
                    self.tb_writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
                    self.tb_writer.add_scalar('Epoch', epoch + 1, global_step)

                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'train_loss': batch_loss,
                        'train_rmse': batch_metrics['rmse'],
                        'train_mae': batch_metrics['mae'],
                        'train_psnr': batch_metrics['psnr'],
                        'train_ssim': batch_metrics['ssim'],
                        'train_msssim': batch_metrics['msssim'],
                        'train_corr2': batch_metrics['corr2'],
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch + 1
                    })
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    self.optimizer.zero_grad()
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise exception

    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        running_metrics = {'rmse': 0.0, 'mae': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'msssim': 0.0, 'corr2': 0.0}
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["img"].to(self.device)
                prior = batch["prior"].to(self.device)
                labels = batch["label"].to(self.device)

                prediction = self.model(images, prior)
                loss = self.criterion(prediction, labels)

                running_loss += loss.item()

                # Calculate metrics
                batch_metrics = self.calculate_metrics(prediction, labels)
                for key in running_metrics:
                    running_metrics[key] += batch_metrics[key]

        # Calculate averages
        avg_loss = running_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in running_metrics.items()}

        print(
            f'[Epoch {epoch + 1}] LR: {self.optimizer.param_groups[0]["lr"]}, Val_Loss: {avg_loss:.6f}, Val_RMSE: {avg_metrics["rmse"]:.6f}, '
            f'Val_MAE: {avg_metrics["mae"]:.6f}, Val_PSNR: {avg_metrics["psnr"]:.6f}, Val_SSIM: {avg_metrics["ssim"]:.6f}, Val_MSSSIM: {avg_metrics["msssim"]:.6f}, Val_CORR2: {avg_metrics["corr2"]:.6f}')

        # 检查是否是当前最佳性能
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_val_metrics = avg_metrics
            self.best_epoch = epoch + 1
            # 模型保存目录
            model_save_dir = os.path.join(self.experiment_dir, TRAINING_CONFIG['model_save_dir'])
            os.makedirs(model_save_dir, exist_ok=True)

            # Save best model
            if epoch > 1:
                save_model(
                    self.model,
                    epoch + 1,
                    avg_loss,
                    model_save_dir,
                    'nnet_medical_ct_best'
                )
                # 伪影可视化
                if self.tb_writer:
                    # 可视化伪影去除效果
                    # 将数据移动到CPU并转换为浮点型
                    img_cpu = images[0].cpu().float()
                    prior_cpu = prior[0].cpu().float()
                    label_cpu = labels[0].cpu().float()
                    pred_cpu = prediction[0].cpu().float()

                    grid = torchvision.utils.make_grid(
                        [img_cpu, prior_cpu, label_cpu, pred_cpu],
                        nrow=4,
                        normalize=True
                    )
                    self.tb_writer.add_image(f"Validation/Artifact_Removal", grid, epoch)

        # 打印当前最佳结果
        print(f'[Best Validation] Epoch: {self.best_epoch}, Loss: {self.best_val_loss:.6f}, '
              f'RMSE: {self.best_val_metrics["rmse"]:.6f}, MAE: {self.best_val_metrics["mae"]:.6f}, '
              f'PSNR: {self.best_val_metrics["psnr"]:.6f}, SSIM: {self.best_val_metrics["ssim"]:.6f}, MSSSIM: {self.best_val_metrics["msssim"]:.6f}, CORR2: {self.best_val_metrics["corr2"]:.6f}')
        # Log to tensorboard
        if self.tb_writer:
            self.tb_writer.add_scalar('Val/Loss', avg_loss, epoch)
            self.tb_writer.add_scalar('Val/RMSE', avg_metrics['rmse'], epoch)
            self.tb_writer.add_scalar('Val/MAE', avg_metrics['mae'], epoch)
            self.tb_writer.add_scalar('Val/PSNR', avg_metrics['psnr'], epoch)
            self.tb_writer.add_scalar('Val/SSIM', avg_metrics['ssim'], epoch)
            self.tb_writer.add_scalar('Val/MSSSIM', avg_metrics['msssim'], epoch)
            self.tb_writer.add_scalar('Val/CORR2', avg_metrics['corr2'], epoch)

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
                'val_rmse': avg_metrics['rmse'],
                'val_mae': avg_metrics['mae'],
                'val_psnr': avg_metrics['psnr'],
                'val_ssim': avg_metrics['ssim'],
                'msssim': avg_metrics['msssim'],
                'val_corr2': avg_metrics['corr2'],
            })

    def train(self):
        """Main training loop."""
        print("Starting training...")

        for epoch in range(TRAINING_CONFIG['epochs']):
            print(f"\nEpoch {epoch + 1}/{TRAINING_CONFIG['epochs']}")
            print("-" * 50)
            # Training
            self.train_epoch(epoch)
            # Validation
            self.validate_epoch(epoch)

        self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        if self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb:
            wandb.finish()

def save_configs(experiment_dir):
    """保存所有配置到实验目录"""
    config_dir = os.path.join(experiment_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    # 保存训练配置
    with open(os.path.join(config_dir, "training_config.txt"), "w") as f:
        f.write("Training Configuration:\n")
        for key, value in TRAINING_CONFIG.items():
            f.write(f"{key}: {value}\n")

    # 保存数据集配置
    with open(os.path.join(config_dir, "dataset_config.txt"), "w") as f:
        f.write("Dataset Configuration:\n")
        for key, value in DATASET_CONFIG.items():
            f.write(f"{key}: {value}\n")

    # 保存模型配置
    with open(os.path.join(config_dir, "model_config.txt"), "w") as f:
        f.write("Model Configuration:\n")
        for key, value in MODEL_CONFIG.items():
            f.write(f"{key}: {value}\n")

    # 保存日志配置
    with open(os.path.join(config_dir, "logging_config.txt"), "w") as f:
        f.write("Logging Configuration:\n")
        for key, value in LOGGING_CONFIG.items():
            f.write(f"{key}: {value}\n")

    # 保存调度器配置
    with open(os.path.join(config_dir, "scheduler_config.txt"), "w") as f:
        f.write("Scheduler Configuration:\n")
        for key, value in SCHEDULER_CONFIG.items():
            f.write(f"{key}: {value}\n")

    # 保存设备配置
    with open(os.path.join(config_dir, "device_config.txt"), "w") as f:
        f.write("Device Configuration:\n")
        for key, value in DEVICE_CONFIG.items():
            f.write(f"{key}: {value}\n")

    print(f"All configurations saved to: {config_dir}")


def copy_current_script(experiment_dir):
    """复制当前训练脚本到实验目录的configs文件夹"""
    # 获取当前脚本的路径
    current_script_path = os.path.abspath(__file__)

    # 创建目标路径
    config_dir = os.path.join(experiment_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    # 构建目标文件路径
    script_filename = os.path.basename(current_script_path)
    dest_path = os.path.join(config_dir, script_filename)

    # 复制文件
    shutil.copy2(current_script_path, dest_path)
    print(f"Current training script copied to: {dest_path}")

def main():
    """Main function."""
    print("PyTorch version:", torch.__version__)
    # 设置确定性训练（可复现性）
    set_determinism(seed=TRAINING_CONFIG['seed'])

    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("experiments", "Nnet", f"{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created experiment directory: {experiment_dir}")

    # 保存当前配置到实验目录
    save_configs(experiment_dir)

    # 复制当前训练脚本
    copy_current_script(experiment_dir)

    # Initialize and start training
    trainer = NnetTrainer(experiment_dir)
    trainer.train()

    print("N-net training completed successfully!")


if __name__ == "__main__":
    main()