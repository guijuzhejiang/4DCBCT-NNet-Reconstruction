import torch
from monai.metrics import CumulativeIterationMetric


class Corr2Metric(CumulativeIterationMetric):
    """
    2次元相関係数を計算するカスタムメトリッククラス
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.corr_values = []  # 相関係数値を格納するためのリスト

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        バッチ処理された2次元相関係数を計算する（純粋なPyTorch実装）
        @param {torch.Tensor} y_pred - 予測テンソル
        @param {torch.Tensor} y - ターゲットテンソル
        @returns {torch.Tensor} - バッチごとの相関係数
        """
        # 入力が浮動小数点型であることを確認
        y_pred = y_pred.float()
        y = y.float()

        # 空間次元をフラット化（バッチとチャネル次元は保持）
        # [B, C, H, W] -> [B, C, H*W]
        pred_flat = y_pred.flatten(start_dim=2)
        target_flat = y.flatten(start_dim=2)

        # 各チャネルの相関係数を計算
        batch_results = []
        for b in range(y_pred.shape[0]):  # バッチを反復処理
            channel_results = []
            for c in range(y_pred.shape[1]):  # チャネルを反復処理
                pred_channel = pred_flat[b, c]
                target_channel = target_flat[b, c]

                # 平均を計算
                pred_mean = pred_channel.mean()
                target_mean = target_channel.mean()

                # 共分散を計算
                covariance = ((pred_channel - pred_mean) *
                              (target_channel - target_mean)).sum()

                # 標準偏差を計算
                pred_std = torch.sqrt(((pred_channel - pred_mean) ** 2).sum())
                target_std = torch.sqrt(((target_channel - target_mean) ** 2).sum())

                # 相関係数を計算（ゼロ除算防止のためにepsilonを追加）
                eps = 1e-8
                corr = covariance / (pred_std * target_std + eps)
                channel_results.append(corr)

            # バッチ内のチャネル平均を計算
            batch_results.append(torch.stack(channel_results).mean())

        # バッチ結果を返す [B]
        return torch.stack(batch_results)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        """結果を正しく蓄積するために親クラスのメソッドをオーバーライド
        @param {torch.Tensor} y_pred - 予測テンソル
        @param {torch.Tensor} y - ターゲットテンソル
        """
        # 現在のバッチの相関係数を計算
        batch_corr = self._compute_tensor(y_pred, y)

        # 後続の集計のために結果を格納
        self.corr_values.append(batch_corr)

        # MONAIフレームワークとの互換性を維持するために親クラスのメソッドを呼び出す
        return super().__call__(y_pred, y)

    def aggregate(self):
        """
        すべてのバッチの結果を集計する
        @returns {torch.Tensor} - 集計された相関係数
        """
        if not self.corr_values:
            return torch.tensor(0.0)

        # すべてのバッチの相関係数を結合
        all_corr = torch.cat(self.corr_values)

        # 指定された削減方法を適用
        if self.reduction == "mean":
            return all_corr.mean()
        elif self.reduction == "sum":
            return all_corr.sum()
        elif self.reduction == "none":
            return all_corr
        else:
            raise ValueError(f"サポートされていない削減モード: {self.reduction}")

    def reset(self):
        """蓄積された状態をリセットする"""
        super().reset()
        self.corr_values = []