import torch
from monai.metrics import CumulativeIterationMetric


class Corr2Metric(CumulativeIterationMetric):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.corr_values = []  # 用于存储相关系数值

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        计算批处理的二维相关系数 (纯PyTorch实现)
        """
        # 确保输入是浮点类型
        y_pred = y_pred.float()
        y = y.float()

        # 展平空间维度 (保留批次和通道维度)
        # [B, C, H, W] -> [B, C, H*W]
        pred_flat = y_pred.flatten(start_dim=2)
        target_flat = y.flatten(start_dim=2)

        # 计算每个通道的相关系数
        batch_results = []
        for b in range(y_pred.shape[0]):  # 遍历批次
            channel_results = []
            for c in range(y_pred.shape[1]):  # 遍历通道
                pred_channel = pred_flat[b, c]
                target_channel = target_flat[b, c]

                # 计算均值
                pred_mean = pred_channel.mean()
                target_mean = target_channel.mean()

                # 计算协方差
                covariance = ((pred_channel - pred_mean) *
                              (target_channel - target_mean)).sum()

                # 计算标准差
                pred_std = torch.sqrt(((pred_channel - pred_mean) ** 2).sum())
                target_std = torch.sqrt(((target_channel - target_mean) ** 2).sum())

                # 计算相关系数 (添加epsilon防止除零)
                eps = 1e-8
                corr = covariance / (pred_std * target_std + eps)
                channel_results.append(corr)

            # 计算批次内通道平均
            batch_results.append(torch.stack(channel_results).mean())

        # 返回批次结果 [B]
        return torch.stack(batch_results)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        """覆盖父类方法以正确累积结果"""
        # 计算当前批次的相关系数
        batch_corr = self._compute_tensor(y_pred, y)

        # 存储结果用于后续聚合
        self.corr_values.append(batch_corr)

        # 调用父类方法保持MONAI框架兼容性
        return super().__call__(y_pred, y)

    def aggregate(self):
        """
        聚合所有批次的结果
        """
        if not self.corr_values:
            return torch.tensor(0.0)

        # 连接所有批次的相关系数
        all_corr = torch.cat(self.corr_values)

        # 应用指定的reduction方法
        if self.reduction == "mean":
            return all_corr.mean()
        elif self.reduction == "sum":
            return all_corr.sum()
        elif self.reduction == "none":
            return all_corr
        else:
            raise ValueError(f"不支持的reduction模式: {self.reduction}")

    def reset(self):
        """重置累积状态"""
        super().reset()
        self.corr_values = []