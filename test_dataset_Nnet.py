import skimage.io as io
import glob, os
import torch.utils.data as data
from torchvision.transforms import transforms as T
import numpy as np
from monai.transforms import (Compose, EnsureChannelFirst, Lambda, ToTensor)
import torch

class Test_Dataset(data.Dataset):
    """
    テストデータセットクラス。医療画像ファイル (.img) を読み込み、前処理を行う。
    """
    def __init__(self, root_data: str, fov_folder: str, indices: list[int]):
        """
        データセットを初期化する。
        @param root_data: 画像のルートディレクトリ
        @param indices: データセットに含める被験者 (subject) のインデックスリスト
        """
        self.transform = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            Lambda(func=lambda x: (x.astype(np.float32) - 2000.0) / 3000.0),
            # テンソルに変換
            ToTensor()
        ])
        self.TestSet = []
        print(f"データセットの構築を開始します。ベースパス: {root_data}")

        # すべてのFOVを反復処理
        # for fov_folder in ["FovL", "FovS_180", "FovS_360"]:
        fov_path = os.path.join(root_data, fov_folder)

        # すべての被験者 (subject) を反復処理
        subjects = sorted(
            [d for d in os.listdir(fov_path)
             if os.path.isdir(os.path.join(fov_path, d)) and int(d.split("_")[1]) in indices],
            key=lambda d: int(d.split("_")[1])
        )
        print(f"{fov_folder} 被験者: {subjects}")

        for subject in subjects:
            subject_path = os.path.join(fov_path, subject)

            # 事前情報 (prior) 画像パスを取得 (すべてのフェーズで共有)
            prior_dir = os.path.join(subject_path, "prior")
            prior_files = sorted(glob.glob(os.path.join(prior_dir, "*.img")))

            # すべてのフェーズ (00-04) を反復処理
            # for phase in [f"phase_{i:02d}" for i in range(5)]:
            phase_path = os.path.join(subject_path, 'phase_00')

            # imgディレクトリの存在を確認
            img_dir = os.path.join(phase_path, "img")
            gt_dir = os.path.join(phase_path, "gt")

            if os.path.exists(img_dir) and os.path.exists(gt_dir):
                # このフェーズのすべての画像スライスを取得
                img_files = sorted(glob.glob(os.path.join(img_dir, "*.img")))
                gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.img")))
                # 各img_fileに対して[img_file, prior_file]のペアを作成
                for img_file, prior_file, gt_file in zip(img_files, prior_files, gt_files):
                    self.TestSet.append([img_file, prior_file, gt_file])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        指定されたインデックスのアイテムを取得する。
        @param index: アイテムのインデックス
        @returns: 画像テンソル、事前情報テンソル、画像パスのタプル
        """
        img_path, prior_path, gt_path = self.TestSet[index]
        with open(img_path, 'rb') as f_img:
            raw_dat_img = np.fromfile(f_img, dtype='<i2', count=512*512)
            img = raw_dat_img.reshape((512, 512)).astype(float)

        with open(prior_path, 'rb') as f_prior:
            raw_data_prior = np.fromfile(f_prior, dtype='<i2', count=512*512)
            prior = raw_data_prior.reshape((512, 512)).astype(float)

        with open(gt_path, 'rb') as f_gt:
            raw_data_gt = np.fromfile(f_gt, dtype='<i2', count=512*512)
            gt = raw_data_gt.reshape((512, 512)).astype(float)

        img = self.transform(img)
        prior = self.transform(prior)
        gt = self.transform(gt)

        return img, prior, gt

    def __len__(self) -> int:
        """
        データセットの長さを返す。
        @returns: データセットのアイテム数
        """
        return len(self.TestSet)
