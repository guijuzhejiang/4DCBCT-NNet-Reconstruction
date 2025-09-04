import torch.utils.data as data
import os
import numpy as np
from monai.transforms import Transform
from monai.config import KeysCollection
import glob


class Nnet_Dataset(data.Dataset):
    """
    4D CBCT Hitachiデータセットに適合するDatasetクラス。
    元の.img形式を保持し、HU値を保持した学習をサポート。
    """

    def __init__(self, root_data, fov_type, indices):
        """
        データセットを初期化する。

        @param root_data: 画像ルートディレクトリ
        @param fov_type: 視野タイプ (例: "FovL", "FovS_180", "FovS_360")
        @param indices: 処理するデータセットインデックスのリスト
        """
        self.samples = []
        print(f"データセット構築を開始します。基本パス: {root_data}")

        # すべてのFOVを反復処理
        fov_path = os.path.join(root_data, fov_type)

        # すべての被験者を反復処理
        subjects = [d for d in os.listdir(fov_path)
                    if os.path.isdir(os.path.join(fov_path, d)) and int(d.split("_")[1]) in indices]
        subjects.sort(key=lambda d: int(d.split("_")[1]))
        print(f"{fov_type} 被験者: {subjects}")

        for subject in subjects:
            subject_path = os.path.join(fov_path, subject)

            # 先行画像パスを取得 (すべてのフェーズで共有)
            prior_dir = os.path.join(subject_path, "prior")
            prior_files = sorted(glob.glob(os.path.join(prior_dir, "*.img")))

            # # すべてのフェーズ (00-04) を反復処理
            # for phase in [f"phase_{i:02d}" for i in range(5)]:
            # phase_00のみ使用
            for phase in [f"phase_{i:02d}" for i in range(1)]:
                phase_path = os.path.join(subject_path, phase)

                # imgおよびgtディレクトリが存在するかチェック
                img_dir = os.path.join(phase_path, "img")
                gt_dir = os.path.join(phase_path, "gt")

                if os.path.exists(img_dir) and os.path.exists(gt_dir):
                    # このフェーズのすべての画像とラベルスライスを取得
                    img_files = sorted(glob.glob(os.path.join(img_dir, "*.img")))
                    label_files = sorted(glob.glob(os.path.join(gt_dir, "*.img")))

                    # ファイル数が一致することを確認
                    if len(img_files) != len(label_files) or len(img_files) != len(prior_files):
                        print(f"警告: {fov_type}/{subject}/{phase} ファイル数が一致しません")
                        continue

                    # 各スライスについてサンプルを作成
                    for i in range(len(img_files)):
                        self.samples.append({
                            "img": img_files[i],
                            "prior": prior_files[i],
                            "label": label_files[i]
                        })

        print(f"データセット構築が完了しました。総サンプル数: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError("有効なサンプルが見つかりませんでした。データパスとパラメータを確認してください")

    def __getitem__(self, index):
        """サンプル辞書を返す（パスのみ、実際の読み込みはtransformで完了）"
        @param {int} index - サンプルのインデックス
        @returns {dict} - サンプルを表す辞書
        """
        return self.samples[index]

    def __len__(self):
        """データセットサイズを返す"
        @returns {int} - データセット内のサンプル数
        """
        return len(self.samples)