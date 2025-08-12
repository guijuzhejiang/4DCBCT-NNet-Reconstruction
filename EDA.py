import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage import exposure
from scipy import stats
import joblib
import imageio


sns.set(font='DejaVu Sans')
# 设置全局字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置参数
DATA_ROOT = "/media/zzg/GJ_disk01/data/Medical/4D_Lung_CBCT_Hitachi/dataset"
FOVS = ["FovL", "FovS_180", "FovS_360"]
SAMPLE_SIZE = 500  # 分析的样本数量
OUTPUT_DIR = "./eda_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化结果存储
results = {
    "subject": [],
    "phase": [],
    "slice": [],
    "file_path": [],
    "min_value": [],
    "max_value": [],
    "mean_value": [],
    "std_value": [],
    "kurtosis": [],
    "skewness": [],
    "file_size": [],
    "image_shape": [],
    "unique_values": [],
    "is_constant": [],
    "air_percentage": [],
    "bone_percentage": []
}


def load_ct_slice(file_path):
    """加载CT切片并返回原始HU值数组"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.int16)
        return raw_data.reshape(512, 512)
    except Exception as e:
        print(f"加载失败: {file_path} - {str(e)}")
        return None


def analyze_slice(file_path, subject, phase, slice_idx):
    """分析单个CT切片"""
    # 加载图像
    img = load_ct_slice(file_path)
    if img is None:
        return

    # 基本统计
    min_val = np.min(img)
    max_val = np.max(img)
    mean_val = np.mean(img)
    std_val = np.std(img)

    # 高级统计
    kurtosis = stats.kurtosis(img.flatten())
    skewness = stats.skew(img.flatten())

    # 文件信息
    file_size = os.path.getsize(file_path)

    # 数据质量检查
    unique_vals = len(np.unique(img))
    is_constant = unique_vals == 1

    # 组织分类
    air_mask = img < -500
    bone_mask = img > 300
    air_percentage = np.mean(air_mask) * 100
    bone_percentage = np.mean(bone_mask) * 100

    # 保存结果
    results["subject"].append(subject)
    results["phase"].append(phase)
    results["slice"].append(slice_idx)
    results["file_path"].append(file_path)
    results["min_value"].append(min_val)
    results["max_value"].append(max_val)
    results["mean_value"].append(mean_val)
    results["std_value"].append(std_val)
    results["kurtosis"].append(kurtosis)
    results["skewness"].append(skewness)
    results["file_size"].append(file_size)
    results["image_shape"].append(img.shape)
    results["unique_values"].append(unique_vals)
    results["is_constant"].append(is_constant)
    results["air_percentage"].append(air_percentage)
    results["bone_percentage"].append(bone_percentage)

    return img


def collect_sample_paths():
    """收集要分析的样本路径"""
    sample_paths = []

    for fov in FOVS:
        fov_path = os.path.join(DATA_ROOT, fov)
        subjects = [d for d in os.listdir(fov_path)
                    if os.path.isdir(os.path.join(fov_path, d)) and d.startswith("subject_")]

        for subject in subjects[:3]:  # 每个FOV分析3个受试者
            subject_path = os.path.join(fov_path, subject)

            # 获取所有相位的图像
            for phase in [f"phase_{i:02d}" for i in range(5)]:
                phase_path = os.path.join(subject_path, phase, "img")

                if os.path.exists(phase_path):
                    img_files = sorted(glob.glob(os.path.join(phase_path, "*.img")))

                    # 从每个相位中选择10个切片
                    for slice_idx, file_path in enumerate(img_files[:10]):
                        sample_paths.append({
                            "file_path": file_path,
                            "subject": subject,
                            "phase": phase,
                            "slice_idx": slice_idx
                        })

                        if len(sample_paths) >= SAMPLE_SIZE:
                            return sample_paths

    return sample_paths


def perform_eda():
    """执行完整的EDA分析"""
    print("Start EDA analysis of 4D CBCT dataset...")
    print(f"Number of analyzed samples: {SAMPLE_SIZE}")

    # 收集样本路径
    sample_paths = collect_sample_paths()
    print(f"{len(sample_paths)} sample paths collected")

    # 分析每个切片
    all_images = []
    for sample in tqdm(sample_paths, desc="Analyze CT slices"):
        img = analyze_slice(
            sample["file_path"],
            sample["subject"],
            sample["phase"],
            sample["slice_idx"]
        )
        if img is not None:
            all_images.append(img)

    # 创建数据框
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "ct_slice_stats.csv"), index=False)
    print(f"The statistical results have been saved to {OUTPUT_DIR}/ct_slice_stats.csv")

    return df, all_images


def generate_visualizations(df, all_images):
    """生成数据可视化"""
    print("Generate visualization results...")

    # 1. 基本统计分布
    plt.figure(figsize=(15, 10))

    #最小值分布
    plt.subplot(2, 3, 1)
    sns.histplot(df["min_value"], kde=True, bins=30)
    plt.title("Min Value distribution")
    plt.xlabel("HU")
    #最大值分布
    plt.subplot(2, 3, 2)
    sns.histplot(df["max_value"], kde=True, bins=30)
    plt.title("Max Value distribution")
    plt.xlabel("HU")
    #均值分布
    plt.subplot(2, 3, 3)
    sns.histplot(df["mean_value"], kde=True, bins=30)
    plt.title("Mean Value distribution")
    plt.xlabel("HU")
    #标准差分布
    plt.subplot(2, 3, 4)
    sns.histplot(df["std_value"], kde=True, bins=30)
    plt.title("Standard Deviation Distribution")
    plt.xlabel("HU")
    #空气区域百分比分布
    plt.subplot(2, 3, 5)
    sns.histplot(df["air_percentage"], kde=True, bins=30)
    plt.title("Air Region Percentage Distribution")
    plt.xlabel("%Percentage (%)")
    #骨骼区域百分比分布
    plt.subplot(2, 3, 6)
    sns.histplot(df["bone_percentage"], kde=True, bins=30)
    plt.title("Bone Region Percentage Distribution")
    plt.xlabel("Percentage (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "basic_stats_distribution.png"))
    plt.close()

    # 2. 不同相位的比较
    plt.figure(figsize=(14, 10))

    # 子图1: 不同相位的平均HU值
    ax1 = plt.subplot(2, 2, 1)
    sns.boxplot(x="phase", y="mean_value", data=df, ax=ax1)
    ax1.set_title("Mean HU Value by Respiratory Phase")
    ax1.set_ylabel("Mean HU Value")
    ax1.set_xlabel("Respiratory Phase")

    # 子图2: 不同相位的HU值标准差
    ax2 = plt.subplot(2, 2, 2)
    sns.boxplot(x="phase", y="std_value", data=df, ax=ax2)
    ax2.set_title("HU Standard Deviation by Respiratory Phase")
    ax2.set_ylabel("Standard Deviation")
    ax2.set_xlabel("Respiratory Phase")

    # 子图3: 不同相位的空气区域百分比
    ax3 = plt.subplot(2, 2, 3)
    sns.boxplot(x="phase", y="air_percentage", data=df, ax=ax3)
    ax3.set_title("Air Region Percentage by Respiratory Phase")
    ax3.set_ylabel("Air Percentage (%)")
    ax3.set_xlabel("Respiratory Phase")

    # 子图4: 不同相位的骨骼区域百分比
    ax4 = plt.subplot(2, 2, 4)
    sns.boxplot(x="phase", y="bone_percentage", data=df, ax=ax4)
    ax4.set_title("Bone Region Percentage by Respiratory Phase")
    ax4.set_ylabel("Bone Percentage (%)")
    ax4.set_xlabel("Respiratory Phase")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phase_comparison.png"))
    plt.close()

    # 3. HU值分布直方图
    plt.figure(figsize=(10, 6))
    all_hu = np.concatenate([img.flatten() for img in all_images])
    # 过滤极端值以更好地可视化
    filtered_hu = all_hu[(all_hu >= -1000) & (all_hu <= 1000)]
    sns.histplot(filtered_hu, bins=100, kde=True)
    plt.axvline(-1000, color='r', linestyle='--', label='Air')
    plt.axvline(-500, color='g', linestyle='--', label='Lung Tissue')
    plt.axvline(0, color='b', linestyle='--', label='Water')
    plt.axvline(300, color='purple', linestyle='--', label='Bone')
    plt.title("Overall HU Value Distribution")
    plt.xlabel("HU")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "hu_distribution.png"))
    plt.close()

    # 4. 样本图像展示
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle("CT Images with Different Windowing Settings", fontsize=20)

    sample_indices = np.random.choice(len(all_images), 3, replace=False)
    windows = [
        ("Lung Window", -600, 1500),
        ("Mediastinal Window", 40, 400),
        ("Bone Window", 400, 2000),
        ("Full Range", None, None)
    ]

    for i, idx in enumerate(sample_indices):
        img = all_images[idx]

        for j, (title, level, width) in enumerate(windows):
            ax = axes[i, j]

            if level is not None and width is not None:
                vmin = level - width / 2
                vmax = level + width / 2
                display_img = np.clip(img, vmin, vmax)
                ax.imshow(display_img, cmap='gray', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img, cmap='gray')

            ax.set_title(f"{title}\nSample {idx}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"))
    plt.close()

    # 5. 呼吸相位动态变化
    # 选择具有完整5个相位的受试者
    subject_counts = df.groupby('subject')['phase'].nunique()
    valid_subjects = subject_counts[subject_counts >= 5].index.tolist()

    if valid_subjects:
        subject = valid_subjects[0]
        subject_df = df[df["subject"] == subject]

        # 选择一个切片位置（例如中间切片）
        slice_counts = subject_df.groupby('slice')['phase'].nunique()
        valid_slices = slice_counts[slice_counts >= 5].index.tolist()

        if valid_slices:
            slice_idx = valid_slices[0]
            subject_df = subject_df[subject_df["slice"] == slice_idx]

            if len(subject_df) >= 5:
                fig, axes = plt.subplots(1, 5, figsize=(20, 5))
                fig.suptitle(f"Respiratory Phase Changes for Subject {subject}", fontsize=16)

                # 按相位排序
                subject_df = subject_df.sort_values('phase')

                for i, (_, row) in enumerate(subject_df.iterrows()):
                    if i >= 5:
                        break

                    img = load_ct_slice(row["file_path"])
                    axes[i].imshow(img, cmap='gray', vmin=-600, vmax=400)
                    axes[i].set_title(f"Phase: {row['phase']}")
                    axes[i].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, "respiratory_phase_changes.png"))
                plt.close()

    # 6. 创建GIF展示呼吸运动
    create_respiratory_gif(df)

    print("All visualizations saved")


def create_respiratory_gif(df):
    """创建呼吸运动的GIF动画"""
    print("Creating respiratory motion GIF...")

    # 选择具有完整5个相位的受试者和切片
    subject_counts = df.groupby('subject')['phase'].nunique()
    valid_subjects = subject_counts[subject_counts >= 5].index.tolist()

    if not valid_subjects:
        print("No subjects with all 5 phases found for GIF creation")
        return

    subject = valid_subjects[0]
    subject_df = df[df["subject"] == subject]

    slice_counts = subject_df.groupby('slice')['phase'].nunique()
    valid_slices = slice_counts[slice_counts >= 5].index.tolist()

    if not valid_slices:
        print(f"No valid slices found for subject {subject}")
        return

    slice_idx = valid_slices[0]
    subject_df = subject_df[subject_df["slice"] == slice_idx]

    if len(subject_df) < 5:
        print(f"Not enough phase data for slice {slice_idx} of subject {subject}")
        return

    # 按相位排序
    subject_df = subject_df.sort_values('phase')

    # 加载图像
    images = []
    for _, row in subject_df.iterrows():
        img = load_ct_slice(row["file_path"])
        if img is None:
            continue

        # 应用肺窗
        windowed = np.clip(img, -1000, 400)  # 更宽的窗位以捕捉变化
        normalized = (windowed - (-1000)) / 1400 * 255
        images.append(normalized.astype(np.uint8))

    # 保存为GIF
    gif_path = os.path.join(OUTPUT_DIR, "respiratory_motion.gif")
    imageio.mimsave(gif_path, images, duration=0.5)
    print(f"Respiratory motion GIF saved to {gif_path}")


def generate_summary_report(df):
    """生成EDA摘要报告"""
    print("Generating EDA summary report...")

    report = "# 4D CBCT Dataset EDA Analysis Report\n\n"
    report += f"**Sample Size:** {len(df)}\n\n"

    # 基本统计
    report += "## Basic Statistics\n"
    report += f"- **Average HU Range:** {df['min_value'].mean():.1f} to {df['max_value'].mean():.1f}\n"
    report += f"- **Overall Mean HU:** {df['mean_value'].mean():.1f} +/- {df['mean_value'].std():.1f}\n"
    report += f"- **Overall HU Std Dev:** {df['std_value'].mean():.1f} +/- {df['std_value'].std():.1f}\n"
    report += f"- **Air Region Percentage:** {df['air_percentage'].mean():.1f}% +/- {df['air_percentage'].std():.1f}%\n"
    report += f"- **Bone Region Percentage:** {df['bone_percentage'].mean():.1f}% +/- {df['bone_percentage'].std():.1f}%\n\n"

    # 数据质量问题
    constant_slices = df[df["is_constant"]]
    report += "## Data Quality Issues\n"
    report += f"- **Constant Value Slices:** {len(constant_slices)} ({len(constant_slices) / len(df) * 100:.2f}%)\n"

    if not constant_slices.empty:
        report += "  Examples of constant slices:\n"
        for path in constant_slices["file_path"].head(3):
            report += f"  - {path}\n"

    # 文件大小分析
    report += "\n## File Size Analysis\n"
    report += f"- **Average File Size:** {df['file_size'].mean() / 1024:.1f} KB\n"
    report += f"- **Min File Size:** {df['file_size'].min() / 1024:.1f} KB\n"
    report += f"- **Max File Size:** {df['file_size'].max() / 1024:.1f} KB\n\n"

    # 相位比较
    phase_stats = df.groupby("phase").agg({
        "mean_value": ["mean", "std"],
        "std_value": ["mean", "std"],
        "air_percentage": ["mean", "std"],
    })
    report += "## Statistical Comparison Across Respiratory Phases\n"
    report += phase_stats.to_markdown() + "\n\n"

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "eda_summary_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Summary report saved to {report_path}")


def main():
    """主函数"""
    # 执行EDA分析
    df, all_images = perform_eda()

    # 生成可视化
    generate_visualizations(df, all_images)

    # 生成摘要报告
    generate_summary_report(df)

    print("EDA分析完成！所有结果保存在", OUTPUT_DIR)


if __name__ == "__main__":
    main()