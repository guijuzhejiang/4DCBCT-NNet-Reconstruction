# inference.py
# 学習済みモデルによる推論と定量評価

import os
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import pearsonr
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from test_dataset_Nnet_windows_neg1000_500 import Test_Dataset
from model_Nnet import Nnet
from config import DATASET_CONFIG


# --- ▼▼▼ 設定項目 ▼▼▼ ---
# 学習済みモデルが保存されているフォルダのフルパス指定
OUTPUT_FOLDER = '/home/zzg/data/Medical/20250903_Hitachi_SampleCode/output/20251024_185446'
# --- ▲▲▲ 設定項目 ▲▲▲ ---

def calculate_metrics(gt_img, pred_img):
    # 2つの画像間の評価指標計算
    data_range = float(gt_img.max() - gt_img.min()) if gt_img.size > 0 else 1.0
    if data_range == 0: data_range = 1.0  # フラット画像のゼロ割回避
    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=data_range)
    ssim = structural_similarity(gt_img, pred_img, data_range=data_range)
    corr2, _ = pearsonr(gt_img.flatten(), pred_img.flatten())
    gt_tensor = torch.from_numpy(gt_img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred_img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    msssim = ms_ssim(gt_tensor, pred_tensor, data_range=data_range).item()
    return psnr, ssim, msssim, corr2

def create_boxplot(df, metric, output_folder):
    # ボックスプロットの作成と保存
    plt.figure(figsize=(10, 6))
    
    columns = [f'prior_{metric}', f'noisy_{metric}', f'infer_{metric}']
    labels = ['Prior vs GT', 'Noisy vs GT', 'Restored vs GT']
    df.boxplot(column=columns)
    
    means = {col: df[col].mean() for col in columns}
    
    plt.title(f'{metric.upper()} Comparison (Higher is Better)')
    plt.ylabel(metric.upper())
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 平均値テキストの追加
    for i, col in enumerate(columns):
        plt.text(i + 1, df[col].max(), f'Mean: {means[col]:.4f}', 
                 ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
        
    plt.savefig(os.path.join(output_folder, f'{metric}_boxplot.png'))
    plt.close()

def to_int16_raw(folder_name, img_float):
    # float型配列をint16型のRAWデータに変換
    if folder_name == "noisy":
        a_min = DATASET_CONFIG['ScaleIntensityRange_img_a_min']
        a_max = DATASET_CONFIG['ScaleIntensityRange_img_a_max']
    elif folder_name == "prior":
        a_min = DATASET_CONFIG['ScaleIntensityRange_prior_a_min']
        a_max = DATASET_CONFIG['ScaleIntensityRange_prior_a_max']
    elif folder_name in ["restored", "gt"]:
        a_min = DATASET_CONFIG['ScaleIntensityRange_GT_a_min']
        a_max = DATASET_CONFIG['ScaleIntensityRange_GT_a_max']
    b_min = DATASET_CONFIG['ScaleIntensityRange_b_min']
    b_max = DATASET_CONFIG['ScaleIntensityRange_b_max']

    if b_max == b_min:
        raise ValueError("b_max と b_min は等しいので、逆線形変換は不可能です。")

    hu = ((img_float - b_min) / (b_max - b_min)) * (a_max - a_min) + a_min

    return hu.astype(np.int16)

def inference_and_evaluate():
    # --- 1. 初期設定とパス確認 ---
    model_weights_path = os.path.join(OUTPUT_FOLDER, 'best_model.pt')
    if 'YOUR_TRAINED_MODEL_FOLDER' in OUTPUT_FOLDER or not os.path.exists(model_weights_path):
        print("エラー: モデル重みファイルが見つかりません。")
        print(f"パス: {model_weights_path}")
        print("inference.py内の'OUTPUT_FOLDER'を有効なパスに書き換えてください。")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 2. 保存先フォルダ作成 ---
    base_output_folder = os.path.join(OUTPUT_FOLDER, "inference_results")
    comparison_img_folder = os.path.join(base_output_folder, "comparison_images")
    comparison_raw_base = os.path.join(base_output_folder, "comparison_raw_images")
    evaluation_folder = os.path.join(base_output_folder, "evaluation_results")
    os.makedirs(comparison_img_folder, exist_ok=True)
    os.makedirs(comparison_raw_base, exist_ok=True)
    os.makedirs(evaluation_folder, exist_ok=True)
    print(f"推論・評価結果は '{base_output_folder}' 以下に保存。")

    # --- 3. データセットとDataLoader準備 ---
    test_dataset = Test_Dataset(
        DATASET_CONFIG['data_root'],
        DATASET_CONFIG['test_fov_type'],
        DATASET_CONFIG['test_dataset_indices']
    )
    test_dataloader = DataLoader(
        test_dataset
        , batch_size=1
        , shuffle=False
        , num_workers=0
    )
    if not test_dataset:
        print("エラー: テストデータが見つかりませんでした。")
        return
        
    # --- 4. モデル準備 ---
    model = Nnet()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 5. 推論と評価ループ ---
    evaluation_results = []
    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Inferencing & Evaluating")
        for i, (noisy_img, prior_img, gt_img) in progress_bar:
            noisy_img, prior_img = noisy_img.to(device), prior_img.to(device)
            infer_img_tensor = model(noisy_img, prior_img)

            noisy_np = noisy_img.cpu().numpy().squeeze()
            infer_np = infer_img_tensor.cpu().numpy().squeeze()
            gt_np = gt_img.numpy().squeeze()
            prior_np = prior_img.cpu().numpy().squeeze()

            # ファイル識別子情報の取得
            original_path = test_dataset.TestSet[i][0]
            path_parts = original_path.replace('\\', '/').split('/')
            subject_name = path_parts[-4]
            phase_name = path_parts[-3]
            file_name_stem = os.path.splitext(path_parts[-1])[0]
            file_identifier = f"{subject_name}_{phase_name}_{file_name_stem}"

            # --- 定量評価 ---
            prior_metrics = calculate_metrics(gt_np, prior_np)
            noisy_metrics = calculate_metrics(gt_np, noisy_np)
            infer_metrics = calculate_metrics(gt_np, infer_np)

            evaluation_results.append({
                'filename': file_identifier,
                'prior_psnr': prior_metrics[0], 'noisy_psnr': noisy_metrics[0], 'infer_psnr': infer_metrics[0],
                'prior_ssim': prior_metrics[1], 'noisy_ssim': noisy_metrics[1], 'infer_ssim': infer_metrics[1],
                'prior_msssim': prior_metrics[2], 'noisy_msssim': noisy_metrics[2], 'infer_msssim': infer_metrics[2],
                'prior_corr2': prior_metrics[3], 'noisy_corr2': noisy_metrics[3], 'infer_corr2': infer_metrics[3],
            })
            
            # --- 比較PNG画像の保存 (Prior | Noisy | Restored | GT) ---
            merged_img = np.concatenate((prior_np, noisy_np, infer_np, gt_np), axis=1)
            plt.imsave(os.path.join(comparison_img_folder, f"{file_identifier}_comparison.png"), merged_img, cmap='gray')

            # --- 比較RAWデータの保存 ---
            subject_raw_base = os.path.join(comparison_raw_base, subject_name)
            folders = {"prior": prior_np, "noisy": noisy_np, "restored": infer_np, "gt": gt_np}
            raw_filename = f"{file_name_stem}.img"
            
            for folder_name, image_data in folders.items():
                save_dir = os.path.join(subject_raw_base, folder_name)
                os.makedirs(save_dir, exist_ok=True)
                raw_data = to_int16_raw(folder_name, image_data)
                raw_data.tofile(os.path.join(save_dir, raw_filename))
            
    # --- 6. 評価結果の集計と保存 ---
    if not evaluation_results:
        print("評価結果がありません。処理を終了します。")
        return

    df_scores = pd.DataFrame(evaluation_results)
    df_scores.to_csv(os.path.join(evaluation_folder, 'evaluation_scores.csv'), index=False, float_format='%.6f')
    df_summary = df_scores.describe().transpose()
    df_summary.to_csv(os.path.join(evaluation_folder, 'evaluation_summary.csv'), float_format='%.6f')
    
    for metric in ['psnr', 'ssim', 'msssim', 'corr2']:
        create_boxplot(df_scores, metric, evaluation_folder)
        
    gc.collect()
    print("\n--- 完了 ---")
    print(f"推論と評価が完了しました。")
    print(f"平均スコア (vs GT):")
    for name in ['prior', 'noisy', 'infer']:
        print(f"  {name.capitalize()}:")
        for metric in ['psnr', 'ssim', 'msssim', 'corr2']:
            mean_score = df_scores[f'{name}_{metric}'].mean()
            print(f"    {metric.upper()}: {mean_score:.4f}")
    print("-----------")

if __name__ == "__main__":
    inference_and_evaluate()