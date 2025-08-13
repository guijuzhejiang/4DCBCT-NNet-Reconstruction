import torch.utils.data as data
import os
import numpy as np
from monai.transforms import Transform
from monai.config import KeysCollection
import glob

class LoadRawImgSlice(Transform):
    """加载单个.img切片文件并重塑为2D图像，保留原始HU值"""

    def __init__(self, keys: KeysCollection, shape=(1, 512, 512), dtype=np.int16):
        self.keys = keys
        self.shape = shape
        self.dtype = dtype

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            path = d[key]
            if path.endswith('.img'):
                with open(path, 'rb') as f:
                    raw_data = np.fromfile(f, dtype=self.dtype)
                # 重塑为2D图像并保留原始CT值
                d[key] = raw_data.reshape(self.shape)
        return d


class Nnet_Dataset(data.Dataset):
    """
    适配4D CBCT Hitachi数据集的Dataset类
    保留原始.img格式，支持保留HU值的训练
    """

    def __init__(self, root_data, indices):
        """
        初始化数据集

        参数:
        root_data: 图像根目录
        HMIndex: 数据集索引列表
        """
        self.samples = []
        print(f"开始构建数据集，基础路径: {root_data}")

        # 遍历所有FOV
        for fov in ["FovL", "FovS_180", "FovS_360"]:
            fov_path = os.path.join(root_data, fov)

            # 遍历所有subject
            subjects = [d for d in os.listdir(fov_path)
                        if os.path.isdir(os.path.join(fov_path, d)) and int(d.split("_")[1]) in indices]
            print(f"{fov} subjects: {subjects}")

            for subject in subjects:
                subject_path = os.path.join(fov_path, subject)

                # 获取先验图像路径 (所有相位共享)
                prior_dir = os.path.join(subject_path, "prior")
                prior_files = sorted(glob.glob(os.path.join(prior_dir, "*.img")))

                # 遍历所有phase (00-04)
                for phase in [f"phase_{i:02d}" for i in range(5)]:
                    phase_path = os.path.join(subject_path, phase)

                    # 检查img和gt目录是否存在
                    img_dir = os.path.join(phase_path, "img")
                    gt_dir = os.path.join(phase_path, "gt")

                    if os.path.exists(img_dir) and os.path.exists(gt_dir):
                        # 获取该相位的所有图像和标签切片
                        img_files = sorted(glob.glob(os.path.join(img_dir, "*.img")))
                        label_files = sorted(glob.glob(os.path.join(gt_dir, "*.img")))

                        # 确保文件数量匹配
                        if len(img_files) != len(label_files) or len(img_files) != len(prior_files):
                            print(f"警告: {fov}/{subject}/{phase} 文件数量不匹配")
                            continue

                        # 为每个切片创建样本
                        for i in range(len(img_files)):
                            self.samples.append({
                                "img": img_files[i],
                                "prior": prior_files[i],
                                "label": label_files[i]
                            })

        print(f"数据集构建完成，总样本数: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError("未找到任何有效样本，请检查数据路径和参数")

    def __getitem__(self, index):
        """返回样本字典（仅路径，实际加载在transform中完成）"""
        return self.samples[index]

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)