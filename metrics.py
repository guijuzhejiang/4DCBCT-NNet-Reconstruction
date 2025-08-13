import torch
import numpy as np
from monai.metrics import CumulativeIterationMetric


class Corr2Metric(CumulativeIterationMetric):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.sum = 0.0
        self.count = 0

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        计算批处理的二维相关系数
        """
        batch_corr = 0.0
        batch_size = y_pred.shape[0]

        for i in range(batch_size):
            pred_img = y_pred[i].cpu().numpy().squeeze()
            target_img = y[i].cpu().numpy().squeeze()

            # 使用numpy的corrcoef计算相关系数
            corr_matrix = np.corrcoef(pred_img.flatten(), target_img.flatten())
            batch_corr += corr_matrix[0, 1]

        return batch_corr / batch_size

    def aggregate(self):
        """
        聚合所有批次的结果
        """
        if self.count == 0:
            return torch.tensor(0.0)

        result = self.sum / self.count

        if self.reduction == "mean":
            return torch.tensor(result)
        elif self.reduction == "sum":
            return torch.tensor(self.sum)
        else:
            raise ValueError(f"不支持的reduction模式: {self.reduction}")

    def reset(self):
        self.sum = 0.0
        self.count = 0