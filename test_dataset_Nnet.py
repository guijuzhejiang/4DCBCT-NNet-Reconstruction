import skimage.io as io
import glob, os
import torch.utils.data as data
from torchvision.transforms import transforms as T
import numpy as np
from monai.transforms import (Compose, EnsureChannelFirst, ScaleIntensityRange, ToTensor)


class Test_Dataset(data.Dataset):
    def __init__(self, root_data, indices):
        """
        初始化数据集

        参数:
        root_data: 图像根目录
        HMIndex: 数据集索引列表
        """
        # self.transform = T.ToTensor()
        self.transform = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensityRange(
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # 转换为Tensor
            ToTensor()
        ])
        self.TrainingSet = []
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

                    if os.path.exists(img_dir):
                        # 获取该相位的所有图像和标签切片
                        img_files = sorted(glob.glob(os.path.join(img_dir, "*.img")))
                        # 为每个img_file创建[img_file, prior_file]对
                        for img_file, prior_file in zip(img_files, prior_files):
                            self.TrainingSet.append([img_file, prior_file])

    def __getitem__(self, index):

        img_path, prior_path = self.TrainingSet[index]
        with open(img_path, 'rb') as f_img:
            raw_data_img = np.fromfile(f_img, dtype=np.int16)
            img = raw_data_img.reshape((512, 512)).astype(float)

        with open(prior_path, 'rb') as f_prior:
            raw_data_prior = np.fromfile(f_prior, dtype=np.int16)
            prior = raw_data_prior.reshape((512, 512)).astype(float)

        img = self.transform(img)
        prior = self.transform(prior)

        return img, prior, img_path

    def __len__(self):  # retures the length of the dataset
        return len(self.TrainingSet)
