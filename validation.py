"""
N-net Training Script for Medical CT Data
Refactored version with improved logging, configuration management, and code organization.
"""

import os
import torch
import warnings

warnings.filterwarnings("ignore")

# Import custom modules
from model_Nnet import Nnet
from train_dataset_Nnet import Nnet_Dataset
from config import TRAINING_CONFIG, DATASET_CONFIG, DEVICE_CONFIG
from utils import get_dataset_slice_counts, setup_device
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord)
from monai.data import CacheDataset, ThreadDataLoader
from monai.metrics import SSIMMetric, MAEMetric, PSNRMetric, RMSEMetric, MultiScaleSSIMMetric
import matplotlib.pyplot as plt
from pathlib import Path
from img_reader import CustomIMGReader
from metrics import Corr2Metric


class NnetTrainer:
    """N-net validation class """

    def __init__(self, model_path):
        self.model_path = model_path
        self.device = setup_device(DEVICE_CONFIG['use_cuda'], DEVICE_CONFIG['cuda_device'])
        # 创建结果保存目录
        model_p = Path(model_path).resolve()
        self.result_dir = model_p.parents[1] / 'validation_results'
        os.makedirs(self.result_dir, exist_ok=True)
        # 初始化指标计算器
        self.ssim_metric = SSIMMetric(spatial_dims=2, reduction="mean")
        self.msssim_metric = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, reduction="mean") #因为图像归一化到 [0,1]
        self.mae_metric = MAEMetric(reduction="mean")
        self.psnr_metric = PSNRMetric(max_val=1.0, reduction="mean")
        self.rmse_metric = RMSEMetric(reduction="mean")
        self.corr2_metric = Corr2Metric(reduction="mean")
        self.setup_data()
        self.setup_model()

    def setup_data(self):
        # 训练集transforms - 针对伪影任务的增强
        val_transforms = Compose([
            LoadImaged(keys=["img", "prior", "label"]),
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

        val_indices = DATASET_CONFIG['val_dataset_indices']
        print("val_indices:", val_indices)
        val_dataset = Nnet_Dataset(DATASET_CONFIG['data_root'], val_indices)
        print(f'Total validation data samples: {len(val_dataset)}')

        self.val_dataset = CacheDataset(
            data=val_dataset,
            transform=val_transforms,
            cache_rate=1.0,
            num_workers=8,
            progress=True
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
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

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

    def validate(self):
        """Validate for one epoch."""
        running_metrics = {'rmse': 0.0, 'mae': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'msssim': 0.0, 'corr2': 0.0}
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["img"].to(self.device)
                prior = batch["prior"].to(self.device)
                labels = batch["label"].to(self.device)
                prediction = self.model(images, prior)
                # Calculate metrics
                batch_metrics = self.calculate_metrics(prediction, labels)
                for key in running_metrics:
                    running_metrics[key] += batch_metrics[key]

            num = min(10, images.size(0))
            fig, axes = plt.subplots(num, 4, figsize=(10, 2.5 * num))
            for i in range(num):
                for j, (tensor, title) in enumerate(zip(
                        [images[i], prior[i], labels[i], prediction[i]],
                        ["Input", "Prior", "Label", "Prediction"])):
                    ax = axes[i, j] if num > 1 else axes[j]
                    ax.axis("off")
                    if i == 0:  # 只在第一行显示标题
                        ax.set_title(title)
                    ax.imshow(tensor.cpu().squeeze(), cmap="gray")
            save_path = os.path.join(self.result_dir, f'validation_results.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison image to {save_path}")
            plt.show()
            plt.close(fig)

        # Calculate averages
        avg_metrics = {k: v / len(self.val_loader) for k, v in running_metrics.items()}
        # 创建格式化结果字符串
        formatted_result = (
            f'Val_RMSE: {avg_metrics["rmse"]:.6f}, '
            f'Val_MAE: {avg_metrics["mae"]:.6f}, '
            f'Val_PSNR: {avg_metrics["psnr"]:.6f}, '
            f'Val_SSIM: {avg_metrics["ssim"]:.6f}, '
            f"Val_MSSSIM: {avg_metrics['msssim']:.6f}, "
            f"Val_CORR2: {avg_metrics['corr2']:.6f}"
        )

        # 打印结果到控制台
        print(formatted_result)

        # 保存结果到文件
        metrics_path = os.path.join(self.result_dir, 'validation_metrics.txt')
        with open(metrics_path, 'w') as f:
            # 保存详细指标
            f.write('Detailed Metrics:\n')
            f.write(f"RMSE: {avg_metrics['rmse']:.6f}\n")
            f.write(f"MAE: {avg_metrics['mae']:.6f}\n")
            f.write(f"PSNR: {avg_metrics['psnr']:.6f}\n")
            f.write(f"SSIM: {avg_metrics['ssim']:.6f}\n")
            f.write(f"MSSSIM: {avg_metrics['msssim']:.6f}\n")
            f.write(f"CORR2: {avg_metrics['corr2']:.6f}\n")

        print(f"Saved metrics to: {metrics_path}")

        return avg_metrics

def main():
    # 创建实验目录
    model_path = './experiments/Nnet/20250701_110121/trained_model/nnet_medical_ct_best_epoch50_loss0.0334.pth'
    trainer = NnetTrainer(model_path)
    trainer.validate()

if __name__ == "__main__":
    main()