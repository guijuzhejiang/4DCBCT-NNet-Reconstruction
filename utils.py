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
    各データセットディレクトリ内のPNG画像を自動的にカウントする。

    @param data_root: データセットフォルダを含むルートディレクトリ
    @param indices: 処理するデータセットインデックスのリスト

    @returns: スライス数
    """
    slice_counts = []

    for idx in indices:
        # ディレクトリ名パターンを構築 (例: "100HM10395")
        dir_pattern = f"{idx}HM10395"
        png_files = glob.glob(os.path.join(data_root, dir_pattern, 'DegradePhase1', '*.png'))
        png_count = len(png_files)
        if png_count > 0:
            slice_counts.append(png_count)
            print(f"データセット {idx}: {png_count} 個のPNG画像が見つかりました")
        else:
            slice_counts.append(0)
            print(f"警告: インデックス {idx} のPNGファイルは見つかりませんでした")
    return slice_counts


def setup_device(use_cuda: bool = True, cuda_device: int = 0) -> torch.device:
    """
    学習に適したデバイスを設定し、返す。

    @param use_cuda: CUDAが利用可能な場合に使用するかどうか
    @param cuda_device: CUDAデバイスインデックス

    @returns: torch.deviceオブジェクト
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
        print(f"使用デバイス: {device}")
        print(f"GPU: {torch.cuda.get_device_name(cuda_device)}")
        print(f"メモリ: {torch.cuda.get_device_properties(cuda_device).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("使用デバイス: CPU")

    return device


def save_model(model: torch.nn.Module, epoch: int, loss: float, save_dir: str, model_name: str = "nnet") -> str:
    """
    モデルのチェックポイントを保存する。

    @param model: 保存するモデル
    @param epoch: 現在のエポック
    @param loss: 現在の損失値
    @param save_dir: モデルを保存するディレクトリ
    @param model_name: モデルファイルのベース名

    @returns: 保存されたモデルへのパス
    """
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{model_name}_epoch{epoch}_loss{loss:.6f}.pth"
    filepath = os.path.join(save_dir, filename)

    # 互換性を高めるため、モデル全体ではなくモデルの状態辞書を保存
    torch.save(model.state_dict(), filepath)
    print(f"モデルを保存しました: {filepath}")

    return filepath


def calculate_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    画像品質評価指標を計算する（評価指標として、値が大きいほど良い）
    - PSNR: ピーク信号対雑音比 (高いほど良い)
    - SSIM: 構造的類似性 (高いほど良い)
    - MSE: 平均二乗誤差 (評価指標に変換: 1/(1+MSE))
    - MAE: 平均絶対誤差 (評価指標に変換: 1/(1+MAE))
    @param prediction: 予測画像テンソル
    @param target: ターゲット画像テンソル
    @returns: 計算された指標を含む辞書
    """
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(prediction, target).item()
        mae = torch.nn.functional.l1_loss(prediction, target).item()

        # PSNRを計算
        psnr = 10 * math.log10(1.0 / mse)

        # SSIMを計算
        ssim_val = ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True,
            win_size=11,  # 通常11x11ウィンドウを使用
            win_sigma=1.5  # ガウス核の標準偏差
        ).item()

        return {
            'mse': mse,
            'mae': mae,
            'psnr': psnr,
            'ssim': ssim_val
        }

def free_memory():
    """メモリリソースを強制的に解放する"""
    # Pythonのガベージコレクションを強制
    gc.collect()

    # CUDAキャッシュをクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # メモリの状態を報告
    mem = psutil.virtual_memory()
    print(f"メモリ解放後: 使用済み {mem.used / 1e9:.1f}GB, 利用可能 {mem.available / 1e9:.1f}GB")