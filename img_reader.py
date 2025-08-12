import numpy as np
from typing import Any, Dict, Sequence, Union, Tuple
from monai.data import ImageReader
from monai.utils import ensure_tuple


class CustomIMGReader(ImageReader):
    """
    自定义的IMG文件读取器，用于读取512x512的int16 HU值医疗图像
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
        验证文件后缀是否为.img
        """
        filenames = ensure_tuple(filename)
        for name in filenames:
            if not isinstance(name, str):
                return False
            if not name.lower().endswith('.img'):
                return False
        return True

    def read(self, data: Union[Sequence[str], str], **kwargs) -> Any:
        """读取图像数据"""
        filenames = ensure_tuple(data)
        if len(filenames) > 1:
            raise ValueError("一次只能读取一个文件")

        with open(filenames[0], 'rb') as f:
            raw_data = np.fromfile(f, dtype=self.dtype)
        return raw_data.reshape(self.image_shape).astype(self.output_dtype)

    def get_data(self, img: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        获取图像数据和元数据

        Args:
            img: read方法返回的图像数据

        Returns:
            (image_array, metadata_dict)
        """

        # 创建元数据
        metadata = {
            "spatial_shape": self.image_shape,
            'original_channel_dim': 0,
            'dtype': self.output_dtype.__name__,
            'filename_or_obj': kwargs.get('filename', 'custom_img'),
            'affine': np.eye(4),  # 单位矩阵（无空间变换信息）
            "original_affine": np.eye(4),  # MONAI需要此字段
        }

        return img, metadata