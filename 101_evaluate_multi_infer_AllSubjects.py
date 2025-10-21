# evaluate_multi_infer.py (エラー修正版)
# 複数の推論結果を横断的に評価するスクリプト

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import seaborn as sns

# 定量評価に必要なライブラリ
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from scipy.stats import pearsonr
from pytorch_msssim import ms_ssim
import torch

# --- ▼▼▼ 評価パラメータ設定 ▼▼▼ ---

# 1. 評価対象の推論結果フォルダの情報を設定 (subjectフォルダの親階層を指定)
INFER_FOLDERS = {
    'infer1_EffUnetb0': '/home/zzg/data/Medical/20250903_Hitachi_SampleCode/output/20250902_081500_0002_EffUnetb0_FovL_phase_00/inference_results/comparison_raw_images',
    'infer2_EffUnetb2': '/home/zzg/data/Medical/20250903_Hitachi_SampleCode/output/20250902_180746_0003_EffUnetb2_FovL_phase_00/inference_results/comparison_raw_images',
    'NNet_FovL': '/home/zzg/data/Medical/20250903_Hitachi_SampleCode/output/20251021_103049_FovL_neg160_240/inference_results/comparison_raw_images',
}

# 2. 評価対象の画像No.の範囲
IMG_NUM_RANGE = range(0, 383)  # axl0057 ~ axl0455

# 3. ラインプロファイル評価のパラメータ
LP_IMG_NUM = 256
LP_COORD_Y = 304
LP_X_AXIS_START = 32
LP_X_AXIS_END = 475
LP_VALUE_MIN = -1000
LP_VALUE_MAX = 1000

# 4. 定量評価の画像領域 (ROI)
ROI = {'x_start': 84, 'y_start': 84, 'x_end': 346, 'y_end': 346}

# 5. 出力フォルダのベース名
OUTPUT_CODE_NAME = '0101_MultiSubject_MultiInferEval'

# --- ▲▲▲ 設定はここまで ▲▲▲ ---


def read_raw_image(path, size=512):
    """16-bit signed intのRAW画像を読み込む"""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int16, count=size*size)
    return data.reshape((size, size))

def calculate_metrics(gt_img, pred_img):
    """2つの画像間の評価指標を計算する"""
    data_range = gt_img.max() - gt_img.min()
    if data_range == 0: return np.inf, 1.0, 1.0, 1.0
    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=data_range)
    ssim = structural_similarity(gt_img, pred_img, data_range=data_range)
    corr2, _ = pearsonr(gt_img.flatten(), pred_img.flatten())
    if gt_img.shape[0] < 161 or gt_img.shape[1] < 161: msssim = np.nan
    else:
        gt_tensor = torch.from_numpy(gt_img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred_img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        msssim = ms_ssim(gt_tensor, pred_tensor, data_range=data_range).item()
    return psnr, ssim, msssim, corr2

# --- ▼▼▼ ここから修正 ▼▼▼ ---
def create_violin_stripplot(df, output_folder, model_order, plot_suffix=""):
    """定量評価の結果をバイオリンプロット+ストリッププロットで保存する"""
    metrics = ['PSNR', 'SSIM', 'MS-SSIM', 'Corr2']
    value_vars_to_melt = [f'{name}_{metric}' for metric in metrics for name in model_order if f'{name}_{metric}' in df.columns]
    df_long = pd.melt(df, id_vars=['filename'], value_vars=value_vars_to_melt, var_name='group', value_name='score')
    df_long[['Model', 'Metric']] = df_long['group'].str.rsplit('_', n=1, expand=True)
    df_long.dropna(inplace=True)
    
    plt.rcParams['font.family'] = 'Times New Roman'
    FONT_SIZE_TITLE = 24; FONT_SIZE_LABEL = 24; FONT_SIZE_TICKS = 18
    plt.rcParams['axes.linewidth'] = 5; plt.rcParams['xtick.major.width'] = 1.5; plt.rcParams['ytick.major.width'] = 1.5
    sns.set_theme(style="whitegrid")
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        subset = df_long[df_long['Metric'] == metric]
        if subset.empty:
            print(f"Warning: {metric} のスコアが全てNaNのため、プロットをスキップします。")
            plt.close()
            continue
            
        # 警告を解消し、hueを指定
        ax = sns.violinplot(data=subset, x='Model', y='score', hue='Model', palette="muted", inner="quartile", order=model_order, legend=False)
        
        # swarmplotからstripplotに変更し、点の重なりを許容
        sns.stripplot(data=subset, x='Model', y='score', color="0.25", size=2.0, alpha=0.3, jitter=True, ax=ax, order=model_order)
        
        ax.set_title(f'{metric} Distribution', fontsize=FONT_SIZE_TITLE)
        ax.set_xlabel('Image Type', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(f'{metric} Score', fontsize=FONT_SIZE_LABEL)
        ax.tick_params(axis='x', labelsize=FONT_SIZE_TICKS, rotation=15)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{metric}_violin_strip{plot_suffix}.png'))
        plt.close()
# --- ▲▲▲ 修正ここまで ▲▲▲ ---

def main():
    """評価処理のメイン関数"""
    # --- 1. 出力フォルダの作成 ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = f'/home/zzg/data/Medical/20250903_Hitachi_SampleCode/output/{timestamp}_{OUTPUT_CODE_NAME}'
    visual_eval_folder = os.path.join(output_folder, 'VisualEvaluation')
    lp_eval_folder = os.path.join(output_folder, 'LineProfileEvaluation')
    quant_eval_folder = os.path.join(output_folder, 'QuantitativeEvaluation')
    os.makedirs(visual_eval_folder, exist_ok=True); os.makedirs(lp_eval_folder, exist_ok=True); os.makedirs(quant_eval_folder, exist_ok=True)
    print(f"評価結果は '{output_folder}' に保存されます。")

    # --- 1.5. 評価対象subjectを自動検出 ---
    subjects_per_infer = [set(os.listdir(p)) for p in INFER_FOLDERS.values() if os.path.isdir(p)]
    if not subjects_per_infer:
        print("エラー: 指定されたINFER_FOLDERSにsubjectフォルダが見つかりません。パスを確認してください。")
        return
    common_subjects = sorted(list(set.intersection(*subjects_per_infer)))
    print(f"共通のsubjectフォルダを検出しました: {common_subjects}")

    base_infer_name = list(INFER_FOLDERS.keys())[0]
    base_infer_path = INFER_FOLDERS[base_infer_name]
    
    quantitative_results = []

    # --- subjectごとのループ処理 ---
    for subject in common_subjects:
        print(f"\n--- Processing Subject: {subject} ---")

        # --- 2. 目視評価用画像の結合と保存 ---
        subject_visual_folder = os.path.join(visual_eval_folder, subject)
        os.makedirs(subject_visual_folder, exist_ok=True)
        for i in tqdm(IMG_NUM_RANGE, desc=f"Visual Eval ({subject})"):
            img_filename = f"axl{i:04d}.img"
            prior_img = read_raw_image(os.path.join(base_infer_path, subject, 'prior', img_filename))
            noisy_img = read_raw_image(os.path.join(base_infer_path, subject, 'noisy', img_filename))
            gt_img = read_raw_image(os.path.join(base_infer_path, subject, 'gt', img_filename))
            if prior_img is None or noisy_img is None or gt_img is None: continue
            
            images_to_merge = [prior_img, noisy_img]
            for name, path in INFER_FOLDERS.items():
                restored_img = read_raw_image(os.path.join(path, subject, 'restored', img_filename))
                if restored_img is not None: images_to_merge.append(restored_img)
            images_to_merge.append(gt_img)
            
            merged_image = np.concatenate(images_to_merge, axis=1)
            merged_image.astype(np.int16).tofile(os.path.join(subject_visual_folder, f"comparison_{i:04d}.img"))

        # --- 3. ラインプロファイル評価 ---
        lp_img_filename = f"axl{LP_IMG_NUM:04d}.img"; lp_data = {}; x_axis_range = np.arange(LP_X_AXIS_START, LP_X_AXIS_END)
        prior_lp_img = read_raw_image(os.path.join(base_infer_path, subject, 'prior', lp_img_filename))
        noisy_lp_img = read_raw_image(os.path.join(base_infer_path, subject, 'noisy', lp_img_filename))
        gt_lp_img = read_raw_image(os.path.join(base_infer_path, subject, 'gt', lp_img_filename))
        if prior_lp_img is not None: lp_data['prior'] = prior_lp_img[LP_COORD_Y, LP_X_AXIS_START:LP_X_AXIS_END]
        if noisy_lp_img is not None: lp_data['noisy'] = noisy_lp_img[LP_COORD_Y, LP_X_AXIS_START:LP_X_AXIS_END]
        if gt_lp_img is not None: lp_data['ground_truth'] = gt_lp_img[LP_COORD_Y, LP_X_AXIS_START:LP_X_AXIS_END]
        for name, path in INFER_FOLDERS.items():
            restored_lp_img = read_raw_image(os.path.join(path, subject, 'restored', lp_img_filename))
            if restored_lp_img is not None: lp_data[name] = restored_lp_img[LP_COORD_Y, LP_X_AXIS_START:LP_X_AXIS_END]
        
        if not lp_data: continue
        df_lp = pd.DataFrame(lp_data, index=x_axis_range); df_lp.index.name = 'Pixel_X'
        df_lp.to_csv(os.path.join(lp_eval_folder, f'line_profile_{subject}_axl{LP_IMG_NUM:04d}.csv'))

        plt.figure(figsize=(15, 8))
        for name, data in lp_data.items():
            style = {'linestyle': '--', 'color': 'green'} if name == 'prior' else {'linestyle': ':', 'color': 'gray'} if name == 'noisy' else {'linewidth': 2, 'color': 'black'} if name == 'ground_truth' else {'alpha': 0.8}
            plt.plot(x_axis_range, data, label=name, **style)
        plt.title(f'Combined Line Profile for {subject} at Y={LP_COORD_Y}', fontsize=16)
        plt.xlabel('Pixel (X-axis)'); plt.ylabel('Pixel Value (HU)'); plt.ylim(LP_VALUE_MIN, LP_VALUE_MAX); plt.grid(True, linestyle='--', alpha=0.6); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(lp_eval_folder, f'line_profile_{subject}_axl{LP_IMG_NUM:04d}_combined.png')); plt.close()

        for name, data in lp_data.items():
            plt.figure(figsize=(15, 8))
            plt.plot(x_axis_range, data, label=name, color='black')
            plt.title(f'Line Profile for {subject} - {name} at Y={LP_COORD_Y}', fontsize=16)
            plt.xlabel('Pixel (X-axis)'); plt.ylabel('Pixel Value (HU)'); plt.ylim(LP_VALUE_MIN, LP_VALUE_MAX); plt.grid(True, linestyle='--', alpha=0.6); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(lp_eval_folder, f'line_profile_{subject}_axl{LP_IMG_NUM:04d}_{name}.png')); plt.close()

        # --- 4. 定量評価 (結果をリストに蓄積) ---
        for i in tqdm(IMG_NUM_RANGE, desc=f"Quant Eval ({subject})"):
            img_filename = f"axl{i:04d}.img"
            gt_img_full = read_raw_image(os.path.join(base_infer_path, subject, 'gt', img_filename))
            if gt_img_full is None: continue
            gt_roi = gt_img_full[ROI['y_start']:ROI['y_end'], ROI['x_start']:ROI['x_end']]
            result_row = {'filename': f"{subject}_{img_filename}"}
            for img_type in ['prior', 'noisy']:
                img_full = read_raw_image(os.path.join(base_infer_path, subject, img_type, img_filename))
                if img_full is not None:
                    img_roi = img_full[ROI['y_start']:ROI['y_end'], ROI['x_start']:ROI['x_end']]
                    psnr, ssim, msssim, corr2 = calculate_metrics(gt_roi, img_roi)
                    result_row[f'{img_type}_PSNR'] = psnr; result_row[f'{img_type}_SSIM'] = ssim
                    result_row[f'{img_type}_MS-SSIM'] = msssim; result_row[f'{img_type}_Corr2'] = corr2
            for name, path in INFER_FOLDERS.items():
                restored_img_full = read_raw_image(os.path.join(path, subject, 'restored', img_filename))
                if restored_img_full is not None:
                    restored_roi = restored_img_full[ROI['y_start']:ROI['y_end'], ROI['x_start']:ROI['x_end']]
                    psnr, ssim, msssim, corr2 = calculate_metrics(gt_roi, restored_roi)
                    result_row[f'{name}_PSNR'] = psnr; result_row[f'{name}_SSIM'] = ssim
                    result_row[f'{name}_MS-SSIM'] = msssim; result_row[f'{name}_Corr2'] = corr2
            quantitative_results.append(result_row)

    # --- 5. 全subjectの定量評価結果を集計・保存 ---
    if not quantitative_results:
        print("\n定量評価の結果がありません。処理を終了します。")
        return

    print("\n--- 全subjectの定量評価結果を集計 ---")
    df_scores = pd.DataFrame(quantitative_results)
    df_scores.to_csv(os.path.join(quant_eval_folder, 'quantitative_scores_all_subjects.csv'), index=False, float_format='%.6f')
    
    model_order_all = ['prior', 'noisy'] + list(INFER_FOLDERS.keys())
    model_order_infer_only = list(INFER_FOLDERS.keys())
    
    # 関数名を変更
    create_violin_stripplot(df_scores, quant_eval_folder, model_order_all, plot_suffix="_all_models")
    create_violin_stripplot(df_scores, quant_eval_folder, model_order_infer_only, plot_suffix="_infer_only")

    print("\n--- 全ての評価が完了しました ---")

if __name__ == '__main__':
    main()