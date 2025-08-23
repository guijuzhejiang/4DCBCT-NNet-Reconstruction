import numpy as np
from typing import Any, Dict, Sequence, Union, Tuple
from monai.data import ImageReader
from monai.utils import ensure_tuple


class CustomIMGReader(ImageReader):
    """
    カスタムIMGファイルリーダー、512x512のint16 HU値医療画像を読み込むために使用
    @param {tuple} image_shape - 画像の形状 (デフォルト: (1, 512, 512))
    @param {np.dtype} dtype - 元の画像データのデータ型 (デフォルト: np.int16)
    @param {type} output_dtype - 出力画像データのデータ型 (デフォルト: float)
    """

    def __init__(self,
                 image_shape: tuple = (1, 512, 512),
                 dtype: np.dtype = np.int16,
                 output_dtype: type = float,
                 **kwargs):
        super().__init__()
        self.image_shape = image_shape
        self.dtype = dtype
        self.output_dtype = output_dtype
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        """
        ファイル拡張子が.imgであるか検証
        @param {Union[Sequence[str], str]} filename - ファイル名またはファイル名のシーケンス
        @returns {bool} - サフィックスが有効な場合はTrue、それ以外はFalse
        """
        filenames = ensure_tuple(filename)
        for name in filenames:
            if not isinstance(name, str):
                return False
            if not name.lower().endswith('.img'):
                return False
        return True

    def read(self, data: Union[Sequence[str], str], **kwargs) -> Any:
        """画像データを読み込む
        @param {Union[Sequence[str], str]} data - 読み込むデータ（ファイルパスまたはパスのシーケンス）
        @returns {Any} - 読み込まれた画像データ
        """
        filenames = ensure_tuple(data)
        if len(filenames) > 1:
            raise ValueError("一度に複数のファイルを読み込むことはできません")

        with open(filenames[0], 'rb') as f:
            raw_data = np.fromfile(f, dtype=self.dtype)
        return raw_data.reshape(self.image_shape).astype(self.output_dtype)

    def get_data(self, img: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        画像データとメタデータを取得

        @param {np.ndarray} img - readメソッドによって返された画像データ
        @returns {Tuple[np.ndarray, Dict]} - (画像配列, メタデータ辞書)
        """

        # メタデータを作成
        metadata = {
            "spatial_shape": self.image_shape,
            'original_channel_dim': 0,
            'dtype': self.output_dtype.__name__,
            'filename_or_obj': kwargs.get('filename', 'custom_img'),
            'affine': np.eye(4),  # 単位行列（空間変換情報なし）
            "original_affine": np.eye(4),  # MONAIで必要
        }

        return img, metadata