"""
医療用CTデータ向けN-net学習スクリプト
ロギング、設定管理、コード構成を改善したリファクタリング版。
"""

import os
import shutil
import numpy as np
import torch
from metrics import Corr2Metric
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision
import warnings

warnings.filterwarnings("ignore")

# カスタムモジュールをインポート
from model_Nnet import Nnet
from train_dataset_Nnet import Nnet_Dataset
from config import TRAINING_CONFIG, DATASET_CONFIG, MODEL_CONFIG, LOGGING_CONFIG, SCHEDULER_CONFIG, DEVICE_CONFIG
from utils import setup_device, save_model
from monai.losses import SSIMLoss, PerceptualLoss, MultiScaleLoss
from torch.nn import MSELoss, L1Loss
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord, LambdaD
)
from monai.data import Dataset, CacheDataset, DataLoader, ThreadDataLoader, SmartCacheDataset, PersistentDataset, LMDBDataset
from monai.metrics import SSIMMetric, MAEMetric, PSNRMetric, RMSEMetric, MultiScaleSSIMMetric
from datetime import datetime
import wandb
from img_reader import CustomIMGReader
from utils import free_memory
from early_stop import EarlyStopping


class NnetTrainer:
    """統合されたロギングとモニタリングを備えたN-netトレーナークラス。
    @param {str} experiment_dir - 実験結果を保存するディレクトリパス。
    """

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.device = setup_device(DEVICE_CONFIG['use_cuda'], DEVICE_CONFIG['cuda_device'])
        self.scheduler_type = SCHEDULER_CONFIG.get('type', 'StepLR')
        print(f"学習率スケジューラ: {self.scheduler_type}を使用します")
        self.fov_type = DATASET_CONFIG.get('train_fov_type', 'FovL')
        # 指標計算器を初期化
        self.ssim_metric = SSIMMetric(spatial_dims=2, reduction="mean")
        self.msssim_metric = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, reduction="mean")
        self.mae_metric = MAEMetric(reduction="mean")
        self.psnr_metric = PSNRMetric(max_val=1.0, reduction="mean")
        self.rmse_metric = RMSEMetric(reduction="mean")
        self.corr2_metric = Corr2Metric(reduction="mean")
        self.setup_logging()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        # 最良の検証性能を追跡する変数
        self.best_val_loss = float('inf')
        self.best_val_metrics = None
        self.best_epoch = 0
        self.last_val_loss = float('inf')
        self.scheduler_lr_epoch = ['StepLR', 'ReduceLROnPlateau']
        self.early_stopping = EarlyStopping(patience=TRAINING_CONFIG.get('early_stopping_patience', 1), verbose=True)

    def setup_logging(self):
        """wandbおよびTensorBoardロギングを設定する。
        """
        # Tensorboard設定
        if LOGGING_CONFIG['use_tensorboard']:
            # 実験ディレクトリ下のサブディレクトリを使用
            tb_log_dir = os.path.join(self.experiment_dir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_log_dir)
            print(f"Tensorboardログは次の場所に保存されます: {tb_log_dir}")
        else:
            self.tb_writer = None

        # Wandb設定
        if LOGGING_CONFIG['use_wandb']:
            # 実験ディレクトリ名をwandbの実行名として使用
            run_name = os.path.basename(self.experiment_dir)
            wandb.init(
                project=LOGGING_CONFIG['wandb_project'],
                entity=LOGGING_CONFIG['wandb_entity'],
                config={
                    **TRAINING_CONFIG,
                    **DATASET_CONFIG,
                    **MODEL_CONFIG,
                    **SCHEDULER_CONFIG
                },
                name=run_name,  # 実行名を設定
                dir=self.experiment_dir  # wandbログを実験ディレクトリに保存するように設定
            )
            self.use_wandb = True
            print("Wandbロギングを初期化しました")
        else:
            self.use_wandb = False
            if LOGGING_CONFIG['use_wandb']:
                print("Wandbが要求されましたが、利用できません")

    # setup_dataメソッド内
    def setup_data(self):
        """データローダを設定する。
        """
        # リーダーを作成
        reader = CustomIMGReader()
        train_val_transforms = Compose([
            LoadImaged(keys=["img", "prior", "label"], reader=reader),
            EnsureChannelFirstd(keys=["img", "prior", "label"]),
            LambdaD(keys=["img", "prior", "label"], func=lambda x: (x.astype(np.float32) - 2000.0) / 3000.0),
            # ScaleIntensityRanged(
            #     keys=["img", "prior", "label"],
            #     a_min=-1000,
            #     a_max=400,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            ToTensord(keys=["img", "prior", "label"])
        ])

        print("データセットを設定中...")

        train_indices = DATASET_CONFIG['train_dataset_indices']
        val_indices = DATASET_CONFIG['val_dataset_indices']

        print("train_indices:", train_indices)
        train_dataset = Nnet_Dataset(DATASET_CONFIG['data_root'], self.fov_type, train_indices)

        print("val_indices:", val_indices)
        val_dataset = Nnet_Dataset(DATASET_CONFIG['data_root'], self.fov_type, val_indices)

        print(f"trainデータセットサイズ: {len(train_dataset)}, 検証データセットサイズ: {len(val_dataset)}")

        # データセットを分割し、変換を適用
        self.train_dataset = Dataset(
            data=train_dataset,
            transform=train_val_transforms,
        )
        self.val_dataset = Dataset(
            data=val_dataset,
            transform=train_val_transforms,
        )

        # MONAI最適化DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=TRAINING_CONFIG['train_batch_size'],
            shuffle=True,
            num_workers=TRAINING_CONFIG['num_workers'],
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
            persistent_workers=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=TRAINING_CONFIG['val_batch_size'],
            num_workers=TRAINING_CONFIG['num_workers'],
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
            persistent_workers=False,
        )

    def setup_model(self):
        """モデルを設定し、デバイスに移動する。
        """
        print("モデルを設定中...")

        self.model = Nnet()
        self.model = self.model.to(self.device)

        # モデルの概要を出力
        print('モデルアーキテクチャ:' + "=" * 60)
        summary(self.model, [(MODEL_CONFIG['input_channels'], *DATASET_CONFIG['image_size']),
                             (MODEL_CONFIG['input_channels'], *DATASET_CONFIG['image_size'])])
        print("=" * 60)

        # モデルをTensorBoardにログ記録
        if self.tb_writer:
            dummy_input1 = torch.randn(1, MODEL_CONFIG['input_channels'], *DATASET_CONFIG['image_size']).to(self.device)
            dummy_input2 = torch.randn(1, MODEL_CONFIG['input_channels'], *DATASET_CONFIG['image_size']).to(self.device)
            self.tb_writer.add_graph(self.model, (dummy_input1, dummy_input2))

    def setup_optimizer(self):
        """オプティマイザと損失関数を設定する。
        """
        batches_per_epoch = len(self.train_loader)
        total_batches = TRAINING_CONFIG['epochs'] * batches_per_epoch
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=SCHEDULER_CONFIG['max_lr']
        )

        # 設定に基づいて学習率スケジューラを選択
        if self.scheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=SCHEDULER_CONFIG['step_size'],
                gamma=SCHEDULER_CONFIG['gamma']
            )
        elif self.scheduler_type == 'ReduceLROnPlateau':         # 改善が停止した際に学習率を低下
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=SCHEDULER_CONFIG['plateau_factor'],
                patience=SCHEDULER_CONFIG['plateau_patience'],
                min_lr=SCHEDULER_CONFIG['ReduceLR_min_lr']
            )
        elif self.scheduler_type == 'CosineAnnealingWarmRestarts':           # ウォームリスタート付きコサインアニーリング
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=SCHEDULER_CONFIG['T_0'] * batches_per_epoch,
                T_mult=SCHEDULER_CONFIG['T_mult'],
            )
        elif self.scheduler_type == 'CosineAnnealingLR':         # コサインアニーリング
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_batches,
            )
        elif self.scheduler_type == 'CyclicLR':          # サイクリック学習率
            self.scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=SCHEDULER_CONFIG['min_lr'],  # ベース学習率
                max_lr=SCHEDULER_CONFIG['max_lr'],  # 最大学習率
                step_size_up=SCHEDULER_CONFIG['epoch_size_up'] * batches_per_epoch,  # ベースから最大までにかかるステップ数
                mode=SCHEDULER_CONFIG['mode']  # 三角モード
            )
        elif self.scheduler_type == 'OneCycleLR':          # ワンサイクル学習率
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=SCHEDULER_CONFIG['max_lr'],
                total_steps=total_batches,
                pct_start=SCHEDULER_CONFIG['pct_start'],
                anneal_strategy='cos',
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=SCHEDULER_CONFIG['T_0'] * batches_per_epoch,        # 初期再起動周期
                T_mult=SCHEDULER_CONFIG['T_mult'],  # 周期乗数
            )

        self.mse_loss = MSELoss().to(self.device)

        # 損失関数を結合
        def combined_loss(pred, target):
            return self.mse_loss(pred, target)

        self.criterion = combined_loss

    def calculate_metrics(self, pred, target):
        """予測とターゲット間の評価指標を計算する。
        @param {torch.Tensor} pred - 予測テンソル。
        @param {torch.Tensor} target - ターゲットテンソル。
        @returns {dict} - 計算された指標を含む辞書。
        """
        with torch.no_grad():
            # 指標をリセット
            self.ssim_metric.reset()
            self.msssim_metric.reset()
            self.mae_metric.reset()
            self.psnr_metric.reset()
            self.rmse_metric.reset()
            self.corr2_metric.reset()

            pred = pred.float()
            target = target.float()

            # 指標を一括計算
            self.ssim_metric(pred, target)
            ssim_val = self.ssim_metric.aggregate().item()

            self.msssim_metric(pred, target)
            msssim_val = self.msssim_metric.aggregate().item()

            self.mae_metric(pred, target)
            mae_val = self.mae_metric.aggregate().item()

            self.psnr_metric(pred, target)
            psnr_val = self.psnr_metric.aggregate().item()

            self.rmse_metric(pred, target)
            rmse_val = self.rmse_metric.aggregate().item()

            self.corr2_metric(pred, target)
            corr2_val = self.corr2_metric.aggregate().item()
            del pred, target
        return {
            "rmse": rmse_val,
            "mae": mae_val,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "msssim": msssim_val,
            "corr2": corr2_val
        }

    def train_epoch(self, epoch):
        """1エポック分の学習を実行する。
        @param {int} epoch - 現在のエポック番号 (0から始まる)。
        """
        self.model.train()
        print(f"\nエポック {epoch + 1}/{TRAINING_CONFIG['epochs']} 学習中")

        for i, batch in enumerate(self.train_loader):
            # batchは辞書形式
            images = batch["img"].to(self.device)       #(batch,1,512,512)
            prior = batch["prior"].to(self.device)      #(batch,1,512,512)
            labels = batch["label"].to(self.device)     #(batch,1,512,512)

            self.optimizer.zero_grad()

            try:
                prediction = self.model(images, prior)       #(batch,1,512,512)
                loss = self.criterion(prediction, labels)
                # 逆伝播
                loss.backward()
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # オプティマイザの更新
                self.optimizer.step()
                # 学習率スケジューラの更新
                if not self.scheduler_type in self.scheduler_lr_epoch:
                    self.scheduler.step()

                if i % 10 == 0:
                    # 損失
                    batch_loss = loss.item()
                    # 指標を計算
                    with torch.no_grad():
                        pred_det = prediction.detach()
                        lbl_det = labels.detach()
                        batch_metrics = self.calculate_metrics(pred_det, lbl_det)

                    print(
                        f"[Epoch {epoch + 1} Batch {i + 1}] "
                        f"LR: {self.optimizer.param_groups[0]['lr']}, "
                        f"Loss: {batch_loss:.6f}, "
                        f"RMSE: {batch_metrics['rmse']:.6f}, "
                        f"MAE: {batch_metrics['mae']:.6f}, "
                        f"PSNR: {batch_metrics['psnr']:.6f}, "
                        f"SSIM: {batch_metrics['ssim']:.6f}, "
                        f"MSSSIM: {batch_metrics['msssim']:.6f}, "
                        f"CORR2: {batch_metrics['corr2']:.6f}"
                    )
                    # TensorBoardにログ記録
                    if self.tb_writer:
                        global_step = epoch * len(self.train_loader) + i
                        self.tb_writer.add_scalar('Train/Loss', batch_loss, global_step)
                        self.tb_writer.add_scalar('Train/RMSE', batch_metrics['rmse'], global_step)
                        self.tb_writer.add_scalar('Train/MAE', batch_metrics['mae'], global_step)
                        self.tb_writer.add_scalar('Train/PSNR', batch_metrics['psnr'], global_step)
                        self.tb_writer.add_scalar('Train/SSIM', batch_metrics['ssim'], global_step)
                        self.tb_writer.add_scalar('Train/MSSSIM', batch_metrics['msssim'], global_step)
                        self.tb_writer.add_scalar('Train/CORR2', batch_metrics['corr2'], global_step)
                        self.tb_writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
                        self.tb_writer.add_scalar('Epoch', epoch + 1, global_step)

                    # Wandbにログ記録
                    if self.use_wandb:
                        wandb.log({
                            'train_loss': batch_loss,
                            'train_rmse': batch_metrics['rmse'],
                            'train_mae': batch_metrics['mae'],
                            'train_psnr': batch_metrics['psnr'],
                            'train_ssim': batch_metrics['ssim'],
                            'train_msssim': batch_metrics['msssim'],
                            'train_corr2': batch_metrics['corr2'],
                            'learning_rate': self.optimizer.param_groups[0]['lr'],
                            'epoch': epoch + 1
                        })
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("警告: メモリ不足")
                    self.optimizer.zero_grad()
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise exception

    def validate_epoch(self, epoch):
        """1エポック分の検証を実行する。
        @param {int} epoch - 現在のエポック番号 (0から始まる)。
        """
        self.model.eval()
        running_loss = 0.0
        running_metrics = {'rmse': 0.0, 'mae': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'msssim': 0.0, 'corr2': 0.0}
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["img"].to(self.device)       #(batch,1,512,512)
                prior = batch["prior"].to(self.device)      #(batch,1,512,512)
                labels = batch["label"].to(self.device)     #(batch,1,512,512)

                prediction = self.model(images, prior)      #(batch,1,512,512)
                loss = self.criterion(prediction, labels)

                running_loss += loss.item()

                # 指標を計算
                batch_metrics = self.calculate_metrics(prediction, labels)
                for key in running_metrics:
                    running_metrics[key] += batch_metrics[key]

        # 平均を計算
        self.last_val_loss = running_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in running_metrics.items()}

        print(
            f"[Epoch {epoch + 1}] "
            f"LR: {self.optimizer.param_groups[0]['lr']}, "
            f"Val_Loss: {self.last_val_loss:.6f}, "
            f"Val_RMSE: {avg_metrics['rmse']:.6f}, "
            f"Val_MAE: {avg_metrics['mae']:.6f}, "
            f"Val_PSNR: {avg_metrics['psnr']:.6f}, "
            f"Val_SSIM: {avg_metrics['ssim']:.6f}, "
            f"Val_MSSSIM: {avg_metrics['msssim']:.6f}, "
            f"Val_CORR2: {avg_metrics['corr2']:.6f}"
        )
        # 現在の最良性能かチェック
        if self.last_val_loss < self.best_val_loss:
            self.best_val_loss = self.last_val_loss
            self.best_val_metrics = avg_metrics
            self.best_epoch = epoch + 1
            # モデル保存ディレクトリ
            model_save_dir = os.path.join(self.experiment_dir, TRAINING_CONFIG['model_save_dir'])
            os.makedirs(model_save_dir, exist_ok=True)

            # 最良モデルを保存
            if epoch > 1:
                save_model(
                    self.model,
                    epoch + 1,
                    self.last_val_loss,
                    model_save_dir,
                    'nnet_medical_ct_best'
                )
                # アーチファクトの可視化
                if self.tb_writer:
                    # アーチファクト除去効果を可視化
                    # データをCPUに移動し、浮動小数点型に変換
                    img_cpu = images[0].cpu().float()       #(1,512,512)
                    prior_cpu = prior[0].cpu().float()      #(1,512,512)
                    label_cpu = labels[0].cpu().float()     #(1,512,512)
                    pred_cpu = prediction[0].cpu().float()  #(1,512,512)

                    grid = torchvision.utils.make_grid(
                        [img_cpu, prior_cpu, label_cpu, pred_cpu],
                        nrow=4,
                        normalize=True
                    )
                    self.tb_writer.add_image(f"Validation/Artifact_Removal", grid, epoch)

        # 現在の最良結果を出力
        print(
            f"[Best Validation] "
            f"Epoch: {self.best_epoch}, "
            f"Loss: {self.best_val_loss:.6f}, "
            f"RMSE: {self.best_val_metrics['rmse']:.6f}, "
            f"MAE: {self.best_val_metrics['mae']:.6f}, "
            f"PSNR: {self.best_val_metrics['psnr']:.6f}, "
            f"SSIM: {self.best_val_metrics['ssim']:.6f}, "
            f"MSSSIM: {self.best_val_metrics['msssim']:.6f}, "
            f"CORR2: {self.best_val_metrics['corr2']:.6f}"
        )
        # TensorBoardにログ記録
        if self.tb_writer:
            self.tb_writer.add_scalar('Val/Loss', self.last_val_loss, epoch)
            self.tb_writer.add_scalar('Val/RMSE', avg_metrics['rmse'], epoch)
            self.tb_writer.add_scalar('Val/MAE', avg_metrics['mae'], epoch)
            self.tb_writer.add_scalar('Val/PSNR', avg_metrics['psnr'], epoch)
            self.tb_writer.add_scalar('Val/SSIM', avg_metrics['ssim'], epoch)
            self.tb_writer.add_scalar('Val/MSSSIM', avg_metrics['msssim'], epoch)
            self.tb_writer.add_scalar('Val/CORR2', avg_metrics['corr2'], epoch)

        # Wandbにログ記録
        if self.use_wandb:
            wandb.log({
                'val_loss': self.last_val_loss,
                'val_rmse': avg_metrics['rmse'],
                'val_mae': avg_metrics['mae'],
                'val_psnr': avg_metrics['psnr'],
                'val_ssim': avg_metrics['ssim'],
                'val_msssim': avg_metrics['msssim'],
                'val_corr2': avg_metrics['corr2'],
            })

        return self.last_val_loss

    def train(self):
        """主要な学習ループ。
        """
        print("学習を開始します...")

        for epoch in range(TRAINING_CONFIG['epochs']):
            print(f"\nEpoch {epoch + 1}/{TRAINING_CONFIG['epochs']}")
            print("-" * 50)
            # 学習
            self.train_epoch(epoch)
            # 検証
            self.validate_epoch(epoch)

            val_loss = self.validate_epoch(epoch)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break

            # メモリを強制的に解放
            free_memory()
            # 学習率スケジューラの更新
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.scheduler.step(self.last_val_loss)
            elif self.scheduler_type == 'StepLR':
                self.scheduler.step()
        self.cleanup()

    def cleanup(self):
        """リソースをクリーンアップする。
        """
        if self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb:
            wandb.finish()

def save_configs(experiment_dir):
    """すべての設定を実験ディレクトリに保存する。
    @param {str} experiment_dir - 実験ディレクトリのパス。
    """
    config_dir = os.path.join(experiment_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    # 学習設定を保存
    with open(os.path.join(config_dir, "training_config.txt"), "w") as f:
        f.write("Training Configuration:\n")
        for key, value in TRAINING_CONFIG.items():
            f.write(f"{key}: {value}\n")

    # データセット設定を保存
    with open(os.path.join(config_dir, "dataset_config.txt"), "w") as f:
        f.write("Dataset Configuration:\n")
        for key, value in DATASET_CONFIG.items():
            f.write(f"{key}: {value}\n")

    # モデル設定を保存
    with open(os.path.join(config_dir, "model_config.txt"), "w") as f:
        f.write("Model Configuration:\n")
        for key, value in MODEL_CONFIG.items():
            f.write(f"{key}: {value}\n")

    # ロギング設定を保存
    with open(os.path.join(config_dir, "logging_config.txt"), "w") as f:
        f.write("Logging Configuration:\n")
        for key, value in LOGGING_CONFIG.items():
            f.write(f"{key}: {value}\n")

    # スケジューラ設定を保存
    with open(os.path.join(config_dir, "scheduler_config.txt"), "w") as f:
        f.write("Scheduler Configuration:\n")
        for key, value in SCHEDULER_CONFIG.items():
            f.write(f"{key}: {value}\n")

    # デバイス設定を保存
    with open(os.path.join(config_dir, "device_config.txt"), "w") as f:
        f.write("Device Configuration:\n")
        for key, value in DEVICE_CONFIG.items():
            f.write(f"{key}: {value}\n")

    print(f"すべての設定を次の場所に保存しました: {config_dir}")


def copy_current_script(experiment_dir):
    """現在の学習スクリプトを実験ディレクトリのconfigsフォルダにコピーする。
    @param {str} experiment_dir - 実験ディレクトリのパス。
    """
    # 現在のスクリプトのパスを取得
    current_script_path = os.path.abspath(__file__)

    # ターゲットパスを作成
    config_dir = os.path.join(experiment_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    # ターゲットファイルパスを構築
    script_filename = os.path.basename(current_script_path)
    dest_path = os.path.join(config_dir, script_filename)

    # ファイルをコピー
    shutil.copy2(current_script_path, dest_path)
    print(f"現在の学習スクリプトを次の場所にコピーしました: {dest_path}")

def main():
    """メイン関数。
    """
    print("PyTorchバージョン:", torch.__version__)
    # 決定論的学習を設定（再現性のため）
    set_determinism(seed=TRAINING_CONFIG['seed'])

    # 実験ディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("experiments", "Nnet", DATASET_CONFIG['train_fov_type'], f"{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"実験ディレクトリを作成しました: {experiment_dir}")

    # 現在の設定を実験ディレクトリに保存
    save_configs(experiment_dir)

    # 現在の学習スクリプトをコピー
    copy_current_script(experiment_dir)

    # 学習を初期化して開始
    trainer = NnetTrainer(experiment_dir)
    trainer.train()

    print("N-net学習が正常に完了しました！")


if __name__ == "__main__":
    main()