"""
Utility functions for N-net training
"""

import os
import glob
from typing import List, Tuple, Dict
import torch
import math
from pytorch_msssim import ssim
import psutil
import gc


def get_dataset_slice_counts(data_root: str, indices: List[int]) -> Tuple[
    List[int], List[int]]:
    """
    Automatically count PNG images in each dataset directory.

    Args:
        data_root: Root directory containing dataset folders
        indices: List of dataset indices to process

    Returns:
        slice_counts
    """
    slice_counts = []

    for idx in indices:
        # Construct directory name pattern (e.g., "100HM10395")
        dir_pattern = f"{idx}HM10395"
        png_files = glob.glob(os.path.join(data_root, dir_pattern, 'DegradePhase1', '*.png'))
        png_count = len(png_files)
        if png_count > 0:
            slice_counts.append(png_count)
            print(f"Dataset {idx}: {png_count} PNG images found")
        else:
            slice_counts.append(0)
            print(f"Warning: No PNG files found for index {idx}")
    return slice_counts


def setup_device(use_cuda: bool = True, cuda_device: int = 0) -> torch.device:
    """
    Setup and return the appropriate device for training.

    Args:
        use_cuda: Whether to use CUDA if available
        cuda_device: CUDA device index

    Returns:
        torch.device object
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(cuda_device)}")
        print(f"Memory: {torch.cuda.get_device_properties(cuda_device).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")

    return device


def save_model(model: torch.nn.Module, epoch: int, loss: float, save_dir: str, model_name: str = "nnet") -> str:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        epoch: Current epoch
        loss: Current loss value
        save_dir: Directory to save model
        model_name: Base name for the model file

    Returns:
        Path to saved model
    """
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{model_name}_epoch{epoch}_loss{loss:.6f}.pth"
    filepath = os.path.join(save_dir, filename)

    # Save model state dict instead of entire model for better compatibility
    torch.save(model.state_dict(), filepath)
    print(f"Model saved: {filepath}")

    return filepath


def calculate_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    计算图像质量评估指标 (作为评估指标，值越大越好)
    - PSNR: 峰值信噪比 (越高越好)
    - SSIM: 结构相似性 (越高越好)
    - MSE: 均方误差 (转换为评估指标: 1/(1+MSE))
    - MAE: 平均绝对误差 (转换为评估指标: 1/(1+MAE))
    """
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(prediction, target).item()
        mae = torch.nn.functional.l1_loss(prediction, target).item()

        # Calculate PSNR
        psnr = 10 * math.log10(1.0 / mse)

        # 计算SSIM
        ssim_val = ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True,
            win_size=11,  # 通常使用11x11窗口
            win_sigma=1.5  # 高斯核标准差
        ).item()

        return {
            'mse': mse,
            'mae': mae,
            'psnr': psnr,
            'ssim': ssim_val
        }

def free_memory():
    """强制释放内存资源"""
    # 强制Python垃圾回收
    gc.collect()

    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 报告内存状态
    mem = psutil.virtual_memory()
    print(f"内存释放后: 已用 {mem.used / 1e9:.1f}GB, 可用 {mem.available / 1e9:.1f}GB")