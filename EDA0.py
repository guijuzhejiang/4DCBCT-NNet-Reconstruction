import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
from config import DATASET_CONFIG


def load_ct_slice(file_path):
    """加载CT切片并返回原始HU值数组"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.int16)
        return raw_data.reshape(512, 512)
    except Exception as e:
        print(f"加载失败: {file_path} - {str(e)}")
        return None


def find_common_subjects_and_slices(data_root, fovs):
    """找到在所有FOV中都存在的受试者和切片"""
    common_data = {}

    for fov in fovs:
        fov_path = os.path.join(data_root, fov)
        if not os.path.exists(fov_path):
            continue

        subjects = [d for d in os.listdir(fov_path)
                    if os.path.isdir(os.path.join(fov_path, d)) and d.startswith("subject_")]

        for subject in subjects:
            subject_path = os.path.join(fov_path, subject)

            # 处理prior文件夹（在subject级别）
            prior_path = os.path.join(subject_path, "prior")
            if os.path.exists(prior_path):
                img_files = sorted(glob.glob(os.path.join(prior_path, "*.img")))
                key = (subject, "prior", "prior")  # 使用特殊的key标识prior
                if key not in common_data:
                    common_data[key] = {}
                common_data[key][fov] = img_files

            # 处理各个phase下的img和gt文件夹
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


def visualize_slice_depth_progression(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    可视化3: 切片深度进展
    固定: subject, phase, image_type
    变化: FOV (行), slice_position (列)
    """
    print("生成切片深度进展矩阵图...")

    common_data = find_common_subjects_and_slices(data_root, fovs)

    # 选择合适的样本
    selected_sample = None
    for (subject, phase, img_type), fov_data in common_data.items():
        if len(fov_data) == len(fovs) and phase == "phase_02" and img_type == "img":
            min_slices = min(len(files) for files in fov_data.values())
            if min_slices > 200:
                selected_sample = (subject, phase, img_type, min_slices)
                break

    if not selected_sample:
        print("未找到合适的样本进行切片深度分析")
        return

    subject, phase, img_type, num_slices = selected_sample

    # 选择不同深度的切片（从头到尾均匀分布）
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
                        ax.imshow(img, cmap='gray', vmin=-1000, vmax=1000)
                        if i == 0:
                            ax.set_title(f"Slice {slice_idx}")
                    else:
                        ax.text(0.5, 0.5, "Load Failed", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "Path Not Found", ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

            # 在第一列添加FOV标签
            if j == 0:
                ax.text(-0.1, 0.5, fov, rotation=90, ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slice_depth_progression.png"), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_subject_comparison(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    可视化4: 不同受试者对比
    固定: FOV, phase, image_type, slice_position
    变化: subject
    """
    print("生成受试者对比矩阵图...")

    # 找到在指定FOV中存在的所有受试者
    fov = "FovL"  # 固定使用FovL
    phase = "phase_02"
    img_type = "img"
    slice_idx = 100

    fov_path = os.path.join(data_root, fov)
    if not os.path.exists(fov_path):
        print(f"FOV路径不存在: {fov_path}")
        return

    subjects = [d for d in os.listdir(fov_path)
                if os.path.isdir(os.path.join(fov_path, d)) and d.startswith("subject_")][:6]  # 最多6个受试者

    if len(subjects) < 2:
        print("可用受试者数量不足")
        return

    # 创建2x3矩阵显示不同受试者
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
                    ax.imshow(img, cmap='gray', vmin=-1000, vmax=1000)
                    ax.set_title(f"{subject}")
                else:
                    ax.text(0.5, 0.5, "Load Failed", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Path Not Found", ha='center', va='center', transform=ax.transAxes)

        ax.axis('off')

    # 隐藏空的子图
    for idx in range(len(subjects), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "subject_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_image_type_comparison(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    可视化5: 图像类型对比 (img vs prior vs gt)
    固定: subject, phase, slice_position
    变化: FOV (行), image_type (列) - 但添加prior列
    """
    print("生成图像类型对比矩阵...")

    common_data = find_common_subjects_and_slices(data_root, fovs)

    # 找到同时有img, gt和prior的样本
    selected_sample = None
    available_subjects = set()

    # 首先收集有img和gt的受试者和相位
    for (subject, phase, img_type), fov_data in common_data.items():
        if img_type in ["img", "gt"] and len(fov_data) == len(fovs):
            available_subjects.add((subject, phase))

    # 然后检查这些受试者是否也有prior数据
    for subject_phase in available_subjects:
        subject, phase = subject_phase
        prior_key = (subject, "prior", "prior")

        if prior_key in common_data and len(common_data[prior_key]) == len(fovs):
            # 检查img和gt是否也存在
            img_key = (subject, phase, "img")
            gt_key = (subject, phase, "gt")

            if (img_key in common_data and len(common_data[img_key]) == len(fovs) and
                    gt_key in common_data and len(common_data[gt_key]) == len(fovs)):
                selected_sample = (subject, phase)
                break

    if not selected_sample:
        print("未找到同时包含img, prior, gt的样本")
        return

    subject, phase = selected_sample
    slice_idx = 100
    img_types = ["img", "prior", "gt"]

    # 创建3x3矩阵 (3个FOV x 3个image_type)
    fig, axes = plt.subplots(len(fovs), len(img_types), figsize=(15, 15))
    fig.suptitle(f"Image Type Comparison\n{subject} - {phase} - Slice {slice_idx}", fontsize=16)

    for i, fov in enumerate(fovs):
        for j, img_type in enumerate(img_types):
            ax = axes[i, j]

            # 构建文件路径 - 根据img_type确定路径
            if img_type == 'prior':
                img_path = os.path.join(data_root, fov, subject, "prior")
            else:
                img_path = os.path.join(data_root, fov, subject, phase, img_type)

            if os.path.exists(img_path):
                img_files = sorted(glob.glob(os.path.join(img_path, "*.img")))
                if slice_idx < len(img_files):
                    img = load_ct_slice(img_files[slice_idx])
                    if img is not None:
                        # 根据图像类型调整显示参数
                        if img_type == "prior":
                            vmin, vmax = -1000, 1000
                        else:
                            vmin, vmax = -1000, 1000

                        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                        if i == 0:
                            ax.set_title(f"{img_type}")
                    else:
                        ax.text(0.5, 0.5, "Load Failed", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "Path Not Found", ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

            # 在第一列添加FOV标签
            if j == 0:
                ax.text(-0.1, 0.5, fov, rotation=90, ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "image_type_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def create_respiratory_motion_gifs(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    创建呼吸运动GIF动画 - 针对每个FOV
    """
    print("创建呼吸运动GIF动画...")

    common_data = find_common_subjects_and_slices(data_root, fovs)

    # 找到有完整相位数据的受试者
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
        print("未找到有完整相位数据的受试者")
        return

    phases = [f"phase_{i:02d}" for i in range(5)]
    slice_idx = 100

    # 为每个FOV创建GIF
    for fov in fovs:
        images = []

        for phase in phases:
            img_path = os.path.join(data_root, fov, selected_subject, phase, "img")
            if os.path.exists(img_path):
                img_files = sorted(glob.glob(os.path.join(img_path, "*.img")))
                if slice_idx < len(img_files):
                    img = load_ct_slice(img_files[slice_idx])
                    if img is not None:
                        # 标准化到0-255范围
                        normalized = np.clip((img + 1000) / 2000 * 255, 0, 255)
                        images.append(normalized.astype(np.uint8))

        if len(images) >= 5:
            # 添加反向播放以形成循环
            images_cycle = images + images[-2:0:-1]

            gif_path = os.path.join(output_dir, f"respiratory_motion_{fov}.gif")
            imageio.mimsave(gif_path, images_cycle, duration=0.3)
            print(f"呼吸运动GIF已保存: {gif_path}")


def visualize_phase_with_prior_matrix(data_root, output_dir, fovs=["FovL", "FovS_180", "FovS_360"]):
    """
    可视化6: 呼吸相位与先验图像对比矩阵
    固定: subject, image_type(img), slice_position
    变化: FOV (行), phase + prior (列)
    """
    print("生成呼吸相位与先验图像对比矩阵...")

    common_data = find_common_subjects_and_slices(data_root, fovs)

    # 找到有完整相位数据和prior数据的受试者
    subject_phase_count = {}
    subjects_with_prior = set()

    # 收集有complete phases的受试者
    for (subject, phase, img_type), fov_data in common_data.items():
        if img_type == "img" and len(fov_data) == len(fovs):
            if subject not in subject_phase_count:
                subject_phase_count[subject] = set()
            subject_phase_count[subject].add(phase)

    # 收集有prior数据的受试者
    for (subject, phase_key, img_type), fov_data in common_data.items():
        if phase_key == "prior" and img_type == "prior" and len(fov_data) == len(fovs):
            subjects_with_prior.add(subject)

    # 选择既有完整相位又有prior的受试者
    selected_subject = None
    for subject, phases in subject_phase_count.items():
        if len(phases) >= 5 and subject in subjects_with_prior:
            selected_subject = subject
            break

    if not selected_subject:
        print("未找到有完整相位和prior数据的受试者")
        return

    phases = [f"phase_{i:02d}" for i in range(5)]
    columns = phases + ["prior"]  # 5个phase + 1个prior
    slice_idx = 100

    # 创建矩阵 (3个FOV x 6列)
    fig, axes = plt.subplots(len(fovs), len(columns), figsize=(24, 12))
    fig.suptitle(f"Respiratory Phases with Prior Reference\n{selected_subject} - img type - Slice {slice_idx}",
                 fontsize=16)

    for i, fov in enumerate(fovs):
        for j, col_item in enumerate(columns):
            ax = axes[i, j]

            # 根据列项确定路径
            if col_item == "prior":
                img_path = os.path.join(data_root, fov, selected_subject, "prior")
            else:  # col_item是phase
                img_path = os.path.join(data_root, fov, selected_subject, col_item, "img")

            if os.path.exists(img_path):
                img_files = sorted(glob.glob(os.path.join(img_path, "*.img")))
                if slice_idx < len(img_files):
                    img = load_ct_slice(img_files[slice_idx])
                    if img is not None:
                        ax.imshow(img, cmap='gray', vmin=-1000, vmax=1000)
                        if i == 0:  # 只在第一行显示列标题
                            ax.set_title(f"{col_item}")
                    else:
                        ax.text(0.5, 0.5, "Load Failed", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "Path Not Found", ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

            # 在第一列添加FOV标签
            if j == 0:
                ax.text(-0.1, 0.5, fov, rotation=90, ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_with_prior_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()


def generate_enhanced_visualizations(data_root, output_dir):
    """
    生成增强的可视化图表
    """
    print("开始生成增强的可视化图表...")

    os.makedirs(output_dir, exist_ok=True)

    fovs = ["FovL", "FovS_180", "FovS_360"]

    try:
        # 1. 切片深度进展
        visualize_slice_depth_progression(data_root, output_dir, fovs)

        # 2. 受试者对比
        visualize_subject_comparison(data_root, output_dir, fovs)

        # 3. 图像类型对比
        visualize_image_type_comparison(data_root, output_dir, fovs)

        # 4. 呼吸相位与先验图像对比矩阵
        visualize_phase_with_prior_matrix(data_root, output_dir, fovs)

        # 5. 创建呼吸运动GIF
        create_respiratory_motion_gifs(data_root, output_dir, fovs)

        print(f"所有增强可视化图表已保存到: {output_dir}")

    except Exception as e:
        print(f"生成可视化时出错: {str(e)}")
        import traceback
        traceback.print_exc()


# 使用示例
if __name__ == "__main__":
    DATA_ROOT = DATASET_CONFIG['data_root']
    OUTPUT_DIR = "./enhanced_eda_results"

    generate_enhanced_visualizations(DATA_ROOT, OUTPUT_DIR)