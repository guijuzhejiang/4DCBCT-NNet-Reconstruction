import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
import seaborn as sns
from config import DATASET_CONFIG


def load_ct_slice(file_path):
    """CTスライスを読み込み、元のHU値配列を返す
    @param {str} file_path - ファイルパス
    @returns {np.ndarray | None} - HU値配列またはNone
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.int16)
        return raw_data.reshape(512, 512)
    except Exception as e:
        print(f"読み込み失敗: {file_path} - {str(e)}")
        return None


def find_common_subjects_and_slices(data_root, fovs):
    """すべてのFOVで存在する被験者とスライスを検索
    @param {str} data_root - データルートディレクトリ
    @param {list} fovs - 視野（FOV）のリスト
    @returns {dict} - 共通データの辞書
    """
    common_data = {}

    for fov in fovs:
        fov_path = os.path.join(data_root, fov)
        if not os.path.exists(fov_path):
            continue

        subjects = [d for d in os.listdir(fov_path)
                    if os.path.isdir(os.path.join(fov_path, d)) and d.startswith("subject_")]

        for subject in subjects:
            subject_path = os.path.join(fov_path, subject)

            # priorフォルダを処理（subjectレベル）
            prior_path = os.path.join(subject_path, "prior")
            if os.path.exists(prior_path):
                img_files = sorted(glob.glob(os.path.join(prior_path, "*.img")))
                key = (subject, "prior", "prior")  # priorを識別する特殊キー
                if key not in common_data:
                    common_data[key] = {}
                common_data[key][fov] = img_files

            # 各phase下のimgとgtフォルダを処理
            for phase in [f"phase_{i:02d}" for i in range(5)]:
                phase_path = os.path.join(subject_path, phase)

                for img_type in ["img", "gt"]:
                    img_path = os.path.join(phase_path, img_type)
                    if os.path.exists(img_path):
                        img_files = sorted(glob.glob(os.path.join(img_path, "*.img")))

                        key = (subject, phase, img_type)
                        if key not in common_data:
                            common_data[key] = {}
                        common_data[key][fov] = img_files

    return common_data


def analyze_hu_distribution(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    各画像タイプのHU値分布を分析し、可視化する
    @param {str} data_root - データルートディレクトリ
    @param {str} output_dir - 出力ディレクトリ
    @param {list} fovs - 視野（FOV）のリスト
    """
    print("HU値分布分析を実行中...")
    
    img_types = ["img", "prior", "gt"]
    hu_data = {img_type: {fov: [] for fov in fovs} for img_type in img_types}
    
    # データ収集
    for fov in fovs:
        fov_path = os.path.join(data_root, fov)
        if not os.path.exists(fov_path):
            print(f"FOVパスが存在しません: {fov_path}")
            continue
            
        subjects = [d for d in os.listdir(fov_path) 
                   if os.path.isdir(os.path.join(fov_path, d)) and d.startswith("subject_")]
        
        # 最初の3つの被験者からサンプル収集（計算量を制限）
        for subject in subjects[:3]:
            subject_path = os.path.join(fov_path, subject)
            
            # prior画像の処理
            prior_path = os.path.join(subject_path, "prior")
            if os.path.exists(prior_path):
                img_files = sorted(glob.glob(os.path.join(prior_path, "*.img")))[:10]  # 最初の10枚
                for img_file in img_files:
                    img = load_ct_slice(img_file)
                    if img is not None:
                        # 全データを使用
                        hu_data["prior"][fov].extend(img.flatten())
            
            # img と gt の処理（phase_00 のみ）
            phase_path = os.path.join(subject_path, "phase_00")
            for img_type in ["img", "gt"]:
                img_path = os.path.join(phase_path, img_type)
                if os.path.exists(img_path):
                    img_files = sorted(glob.glob(os.path.join(img_path, "*.img")))[:10]  # 最初の10枚
                    for img_file in img_files:
                        img = load_ct_slice(img_file)
                        if img is not None:
                            # 全データを使用
                            hu_data[img_type][fov].extend(img.flatten())
    
    # 統計情報の計算と表示
    print("\n=== HU値分布統計 ===")
    stats_data = []
    
    for img_type in img_types:
        for fov in fovs:
            if hu_data[img_type][fov]:
                values = np.array(hu_data[img_type][fov])
                stats = {
                    'Image_Type': img_type,
                    'FOV': fov,
                    'Count': len(values),
                    'Mean': np.mean(values),
                    'Std': np.std(values),
                    'Min': np.min(values),
                    'Max': np.max(values),
                    'Q25': np.percentile(values, 25),
                    'Q50': np.percentile(values, 50),
                    'Q75': np.percentile(values, 75)
                }
                stats_data.append(stats)
                print(f"{img_type} - {fov}: Mean={stats['Mean']:.1f}, Std={stats['Std']:.1f}, "
                      f"Range=[{stats['Min']:.0f}, {stats['Max']:.0f}]")
    
    # 1. ヒストグラム比較（3x3グリッド：img_types × fovs）
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('HU Value Distribution by Image Type and FOV', fontsize=16)
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan']
    
    for i, img_type in enumerate(img_types):
        for j, fov in enumerate(fovs):
            ax = axes[i, j]
            if hu_data[img_type][fov]:
                values = np.array(hu_data[img_type][fov])
                # 全データ範囲でヒストグラムを表示
                ax.hist(values, bins=100, alpha=0.8, color=colors[i*3+j], density=True)
                
                # 統計情報を表示
                mean_val = np.mean(values)
                std_val = np.std(values)
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
                
                ax.set_title(f'{img_type.upper()} - {fov}', fontsize=12)
                ax.set_xlabel('HU Value')
                ax.set_ylabel('Density')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{img_type.upper()} - {fov}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hu_distribution_by_type.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. ボックスプロット比較
    plt.figure(figsize=(15, 8))
    
    # データを準備
    plot_data = []
    for img_type in img_types:
        for fov in fovs:
            if hu_data[img_type][fov]:
                values = np.array(hu_data[img_type][fov])
                # 全データを使用（抽样を削除）
                for val in values:
                    plot_data.append({
                        'HU_Value': val,
                        'Image_Type': img_type,
                        'FOV': fov,
                        'Type_FOV': f"{img_type}_{fov}"
                    })
    
    if plot_data:
        import pandas as pd
        df = pd.DataFrame(plot_data)
        
        plt.subplot(2, 1, 1)
        sns.boxplot(data=df, x='Image_Type', y='HU_Value', hue='FOV')
        plt.title('HU Value Distribution by Image Type and FOV')
        plt.ylabel('HU Value')
        
        plt.subplot(2, 1, 2)
        sns.violinplot(data=df, x='Image_Type', y='HU_Value', hue='FOV')
        plt.title('HU Value Distribution (Violin Plot)')
        plt.ylabel('HU Value')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hu_distribution_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 統計サマリーテーブルの保存
    if stats_data:
        import pandas as pd
        stats_df = pd.DataFrame(stats_data)
        
        # テーブルの可視化
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 数値を小数点1桁に丸める
        display_df = stats_df.copy()
        numeric_cols = ['Mean', 'Std', 'Min', 'Max', 'Q25', 'Q50', 'Q75']
        for col in numeric_cols:
            display_df[col] = display_df[col].round(1)
        
        table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # ヘッダーのスタイル設定
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('HU Value Distribution Statistics Summary', fontsize=16, pad=20)
        plt.savefig(os.path.join(output_dir, "hu_distribution_stats_table.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # CSVファイルとして保存
        stats_df.to_csv(os.path.join(output_dir, "hu_distribution_stats.csv"), index=False)
    
    print(f"HU値分布分析完了。結果は {output_dir} に保存されました。")


def visualize_slice_depth_progression(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    スライス深度進行の可視化
    固定: subject, phase, image_type
    変化: FOV (行), slice_position (列)
    @param {str} data_root - データルートディレクトリ
    @param {str} output_dir - 出力ディレクトリ
    @param {list} fovs - 視野（FOV）のリスト
    """
    print("スライス深度進行行列図を生成中...")

    common_data = find_common_subjects_and_slices(data_root, fovs)

    # 適切なサンプルを選択
    selected_sample = None
    for (subject, phase, img_type), fov_data in common_data.items():
        if len(fov_data) == len(fovs) and phase == "phase_02" and img_type == "img":
            min_slices = min(len(files) for files in fov_data.values())
            if min_slices > 200:
                selected_sample = (subject, phase, img_type, min_slices)
                break

    if not selected_sample:
        print("スライス深度分析に適したサンプルが見つかりませんでした")
        return

    subject, phase, img_type, num_slices = selected_sample

    # 異なる深度のスライスを選択（最初から最後まで均等に分布）
    slice_indices = np.linspace(0, num_slices - 1, 8, dtype=int)

    fig, axes = plt.subplots(len(fovs), len(slice_indices), figsize=(24, 9))
    fig.suptitle(f"Slice Depth Progression\n{subject} - {phase} - {img_type}", fontsize=16)

    for i, fov in enumerate(fovs):
        for j, slice_idx in enumerate(slice_indices):
            ax = axes[i, j]

            img_path = os.path.join(data_root, fov, subject, phase, img_type)
            if os.path.exists(img_path):
                img_files = sorted(glob.glob(os.path.join(img_path, "*.img")))
                if slice_idx < len(img_files):
                    img = load_ct_slice(img_files[slice_idx])
                    if img is not None:
                        ax.imshow(img, cmap='gray')
                        if i == 0:
                            ax.set_title(f"Slice {slice_idx}")
                    else:
                        ax.text(0.5, 0.5, "読み込み失敗", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "データなし", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "パスが見つかりません", ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

            # 最初の列にFOVラベルを追加
            if j == 0:
                ax.text(-0.1, 0.5, fov, rotation=90, ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slice_depth_progression.png"), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_subject_comparison(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    異なる被験者の比較の可視化
    固定: FOV, phase, image_type, slice_position
    変化: subject
    @param {str} data_root - データルートディレクトリ
    @param {str} output_dir - 出力ディレクトリ
    @param {list} fovs - 視野（FOV）のリスト
    """
    print("被験者比較行列図を生成中...")

    # 指定されたFOVに存在するすべての被験者を検索
    fov = "FovL"  # FovLを固定で使用
    phase = "phase_02"
    img_type = "img"

    fov_path = os.path.join(data_root, fov)
    if not os.path.exists(fov_path):
        print(f"FOVパスが存在しません: {fov_path}")
        return

    subjects = [d for d in os.listdir(fov_path)
                if os.path.isdir(os.path.join(fov_path, d)) and d.startswith("subject_")][:6]  # 最大6人の被験者

    if len(subjects) < 2:
        print("利用可能な被験者数が不足しています")
        return

    # 異なる被験者を表示するために2x3の行列を作成
    rows = 2
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle(f"Subject Comparison\n{fov} - {phase} - {img_type} - Slice {slice_idx}", fontsize=16)

    axes_flat = axes.flatten() if len(subjects) > 1 else [axes]

    for idx, subject in enumerate(subjects[:6]):
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]

        img_path = os.path.join(data_root, fov, subject, phase, img_type)
        if os.path.exists(img_path):
            img_files = sorted(glob.glob(os.path.join(img_path, "*.img")))
            if slice_idx < len(img_files):
                img = load_ct_slice(img_files[slice_idx])
                if img is not None:
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"{subject}")
                else:
                    ax.text(0.5, 0.5, "読み込み失敗", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "データなし", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "パスが見つかりません", ha='center', va='center', transform=ax.transAxes)

        ax.axis('off')

    # 空のサブプロットを非表示
    for idx in range(len(subjects), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "subject_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_image_type_comparison(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    画像タイプ比較 (img vs prior vs gt)
    固定: subject, phase, slice_position
    変化: FOV (行), image_type (列) - 但添加prior列
    @param {str} data_root - データルートディレクトリ
    @param {str} output_dir - 出力ディレクトリ
    @param {list} fovs - 視野（FOV）のリスト
    """
    print("画像タイプ比較行列図を生成中...")

    common_data = find_common_subjects_and_slices(data_root, fovs)

    # img, gt, priorをすべて持つサンプルを検索
    selected_sample = None
    available_subjects = set()

    # まずimgとgtを持つ被験者とフェーズを収集
    for (subject, phase, img_type), fov_data in common_data.items():
        if img_type in ["img", "gt"] and len(fov_data) == len(fovs):
            available_subjects.add((subject, phase))

    # 次に、これらの被験者がpriorデータも持っているか確認
    for subject_phase in available_subjects:
        subject, phase = subject_phase
        prior_key = (subject, "prior", "prior")

        if prior_key in common_data and len(common_data[prior_key]) == len(fovs):
            # imgとgtも存在するか確認
            img_key = (subject, phase, "img")
            gt_key = (subject, phase, "gt")

            if (img_key in common_data and len(common_data[img_key]) == len(fovs) and
                    gt_key in common_data and len(common_data[gt_key]) == len(fovs)):
                selected_sample = (subject, phase)
                break

    if not selected_sample:
        print("img, prior, gtをすべて含むサンプルが見つかりませんでした")
        return

    subject, phase = selected_sample
    img_types = ["img", "prior", "gt"]

    # 3x3の行列を作成 (3つのFOV x 3つの画像タイプ)
    fig, axes = plt.subplots(len(fovs), len(img_types), figsize=(15, 15))
    fig.suptitle(f"Image Type Comparison\n{subject} - {phase} - Slice {slice_idx}", fontsize=16)

    for i, fov in enumerate(fovs):
        for j, img_type in enumerate(img_types):
            ax = axes[i, j]

            # ファイルパスを構築 - img_typeに基づいてパスを決定
            if img_type == 'prior':
                img_path = os.path.join(data_root, fov, subject, "prior")
            else:
                img_path = os.path.join(data_root, fov, subject, phase, img_type)

            if os.path.exists(img_path):
                img_files = sorted(glob.glob(os.path.join(img_path, "*.img")))
                if slice_idx < len(img_files):
                    img = load_ct_slice(img_files[slice_idx])
                    if img is not None:
                        # 画像タイプに基づいて表示パラメータを調整
                        # すべてのデータ範囲を表示
                        ax.imshow(img, cmap='gray')
                        if i == 0:
                            ax.set_title(f"{img_type}")
                    else:
                        ax.text(0.5, 0.5, "読み込み失敗", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "データなし", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "パスが見つかりません", ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

            # 最初の列にFOVラベルを追加
            if j == 0:
                ax.text(-0.1, 0.5, fov, rotation=90, ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "image_type_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def create_respiratory_motion_gifs(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    呼吸運動GIFアニメーションの作成 - 各FOVについて
    @param {str} data_root - データルートディレクトリ
    @param {str} output_dir - 出力ディレクトリ
    @param {list} fovs - 視野（FOV）のリスト
    """
    print("呼吸運動GIFアニメーションを作成中...")

    common_data = find_common_subjects_and_slices(data_root, fovs)

    # 完全なフェーズデータを持つ被験者を検索
    subject_phase_count = {}
    for (subject, phase, img_type), fov_data in common_data.items():
        if img_type == "img":
            key = subject
            if key not in subject_phase_count:
                subject_phase_count[key] = set()
            if len(fov_data) == len(fovs):
                subject_phase_count[key].add(phase)

    selected_subject = None
    for subject, phases in subject_phase_count.items():
        if len(phases) >= 5:
            selected_subject = subject
            break

    if not selected_subject:
        print("完全なフェーズデータを持つ被験者が見つかりませんでした")
        return

    phases = [f"phase_{i:02d}" for i in range(5)]

    # 各FOVについてGIFを作成
    for fov in fovs:
        images = []
        for phase in phases:
            img_path = os.path.join(data_root, fov, selected_subject, phase, "img")
            if os.path.exists(img_path):
                img_files = sorted(glob.glob(os.path.join(img_path, "*.img")))
                if slice_idx < len(img_files):
                    img = load_ct_slice(img_files[slice_idx])
                    if img is not None:
                        # 動的範囲に基づいて正規化
                        img_min, img_max = img.min(), img.max()
                        if img_max > img_min:
                            normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            normalized = np.zeros_like(img, dtype=np.uint8)
                        images.append(normalized)

        if len(images) >= 5:
            # ループを形成するために逆再生を追加
            images_cycle = images + images[-2:0:-1]

            gif_path = os.path.join(output_dir, f"respiratory_motion_{fov}.gif")
            imageio.mimsave(gif_path, images_cycle, duration=0.3)
            print(f"呼吸運動GIFを保存しました: {gif_path}")


def visualize_phase_with_prior_matrix(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    呼吸フェーズと先行画像の比較行列の可視化
    固定: subject, image_type(img), slice_position
    変化: FOV (行), phase + prior (列)
    @param {str} data_root - データルートディレクトリ
    @param {str} output_dir - 出力ディレクトリ
    @param {list} fovs - 視野（FOV）のリスト
    """
    print("呼吸フェーズと先行画像比較行列を生成中...")

    common_data = find_common_subjects_and_slices(data_root, fovs)

    # 完全なフェーズデータとpriorデータを持つ被験者を検索
    subject_phase_count = {}
    subjects_with_prior = set()

    # 完全なフェーズを持つ被験者を収集
    for (subject, phase, img_type), fov_data in common_data.items():
        if img_type == "img" and len(fov_data) == len(fovs):
            if subject not in subject_phase_count:
                subject_phase_count[subject] = set()
            subject_phase_count[subject].add(phase)

    # priorデータを持つ被験者を収集
    for (subject, phase_key, img_type), fov_data in common_data.items():
        if phase_key == "prior" and img_type == "prior" and len(fov_data) == len(fovs):
            subjects_with_prior.add(subject)

    # 完全なフェーズとpriorの両方を持つ被験者を選択
    selected_subject = None
    for subject, phases in subject_phase_count.items():
        if len(phases) >= 5 and subject in subjects_with_prior:
            selected_subject = subject
            break

    if not selected_subject:
        print("完全なフェーズデータとpriorデータの両方を持つ被験者が見つかりませんでした")
        return

    phases = [f"phase_{i:02d}" for i in range(5)]
    columns = phases + ["prior"]  # 5つのフェーズ + 1つのprior

    # 行列を作成 (3つのFOV x 6列)
    fig, axes = plt.subplots(len(fovs), len(columns), figsize=(24, 12))
    fig.suptitle(f"Respiratory Phases with Prior Reference\n{selected_subject} - img type - Slice {slice_idx}",
                 fontsize=16)

    for i, fov in enumerate(fovs):
        for j, col_item in enumerate(columns):
            ax = axes[i, j]

            # 列項目に基づいてパスを決定
            if col_item == "prior":
                img_path = os.path.join(data_root, fov, selected_subject, "prior")
            else:  # col_itemがphaseの場合
                img_path = os.path.join(data_root, fov, selected_subject, col_item, "img")

            if os.path.exists(img_path):
                img_files = sorted(glob.glob(os.path.join(img_path, "*.img")))
                if slice_idx < len(img_files):
                    img = load_ct_slice(img_files[slice_idx])
                    if img is not None:
                        ax.imshow(img, cmap='gray')
                        if i == 0:  # 最初の行にのみ列タイトルを表示
                            ax.set_title(f"{col_item}")
                    else:
                        ax.text(0.5, 0.5, "読み込み失敗", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "データなし", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "パスが見つかりません", ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

            # 最初の列にFOVラベルを追加
            if j == 0:
                ax.text(-0.1, 0.5, fov, rotation=90, ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_with_prior_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()


def generate_enhanced_visualizations(data_root, output_dir):
    """
    強化された可視化グラフを生成する
    @param {str} data_root - データルートディレクトリ
    @param {str} output_dir - 出力ディレクトリ
    """
    print("強化された可視化グラフの生成を開始します...")

    os.makedirs(output_dir, exist_ok=True)

    fovs = ["FovL", "FovS_180", "FovS_360"]

    try:
        # 0. HU値分布分析（新規追加）
        analyze_hu_distribution(data_root, output_dir, fovs)

        # 1. スライス深度進行
        # visualize_slice_depth_progression(data_root, output_dir, fovs)
        #
        # # 2. 被験者比較
        # visualize_subject_comparison(data_root, output_dir, fovs)
        #
        # # 3. 画像タイプ比較
        # visualize_image_type_comparison(data_root, output_dir, fovs)
        #
        # # 4. 呼吸フェーズと先行画像比較行列
        # visualize_phase_with_prior_matrix(data_root, output_dir, fovs)
        #
        # # 5. 呼吸運動GIFを作成
        # create_respiratory_motion_gifs(data_root, output_dir, fovs)

        print(f"すべての強化された可視化グラフは次の場所に保存されました: {output_dir}")

    except Exception as e:
        print(f"可視化生成中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()


# 使用例
if __name__ == "__main__":
    DATA_ROOT = DATASET_CONFIG['data_root']
    OUTPUT_DIR = "./eda_results"
    slice_idx = 100 # 全体のslice_idxをここで定義
    generate_enhanced_visualizations(DATA_ROOT, OUTPUT_DIR)