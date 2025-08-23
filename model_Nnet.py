# Nnetアーキテクチャ

import torch  # PyTorchのトップレベルパッケージとテンソルライブラリ
import torch.nn as nn
import torch.nn.functional as F


# モデルアーキテクチャ
class Nnet(nn.Module):
    """
    Nnetモデルの定義。エンコーダ・デコーダ構造とスキップ接続を持つ。
    """

    def ExtractionBlock(self, InChannel_1: int, OutChannel_1: int, Kernal_1: int,
                        InChannel_2: int, OutChannel_2: int, Kernal_2: int) -> nn.Sequential:
        """
        特徴抽出ブロックを構築する。
        @param InChannel_1: 最初の畳み込み層の入力チャネル数
        @param OutChannel_1: 最初の畳み込み層の出力チャネル数
        @param Kernal_1: 最初の畳み込み層のカーネルサイズ
        @param InChannel_2: 2番目の畳み込み層の入力チャネル数
        @param OutChannel_2: 2番目の畳み込み層の出力チャネル数
        @param Kernal_2: 2番目の畳み込み層のカーネルサイズ
        @returns: nn.Sequentialブロック
        """
        DownBlock = nn.Sequential(
            nn.Conv2d(InChannel_1, OutChannel_1, Kernal_1),
            nn.PReLU(),
            nn.Conv2d(InChannel_2, OutChannel_2, Kernal_2),
            nn.PReLU(),
        )
        return DownBlock

    def ExpansionBlock(self, InChannel1: int, OutChannel1: int, Kernel1: int,
                       InChannel2: int, OutChannel2: int, Kernel2: int) -> nn.Sequential:
        """
        特徴拡張ブロックを構築する。
        @param InChannel1: 最初の転置畳み込み層の入力チャネル数
        @param OutChannel1: 最初の転置畳み込み層の出力チャネル数
        @param Kernel1: 最初の転置畳み込み層のカーネルサイズ
        @param InChannel2: 2番目の転置畳み込み層の入力チャネル数
        @param OutChannel2: 2番目の転置畳み込み層の出力チャネル数
        @param Kernel2: 2番目の転置畳み込み層のカーネルサイズ
        @returns: nn.Sequentialブロック
        """
        UpBlock = nn.Sequential(
            nn.ConvTranspose2d(in_channels=InChannel1, out_channels=OutChannel1, kernel_size=Kernel1)
            , nn.PReLU()
            , nn.ConvTranspose2d(in_channels=InChannel2, out_channels=OutChannel2, kernel_size=Kernel2)
            , nn.PReLU()
        )
        return UpBlock

    def FinalConv(self, InChannel_1: int, OutChannel_1: int, Kernal_1: int,
                  InChannel_2: int, OutChannel_2: int, Kernal_2: int,
                  InChannel_3: int, OutChannel_3: int, Kernal_3: int) -> nn.Sequential:
        """
        最終畳み込み層を構築する。
        @param InChannel_1: 最初の畳み込み層の入力チャネル数
        @param OutChannel_1: 最初の畳み込み層の出力チャネル数
        @param Kernal_1: 最初の畳み込み層のカーネルサイズ
        @param InChannel_2: 2番目の畳み込み層の入力チャネル数
        @param OutChannel_2: 2番目の畳み込み層の出力チャネル数
        @param Kernal_2: 2番目の畳み込み層のカーネルサイズ
        @param InChannel_3: 3番目の畳み込み層の入力チャネル数
        @param OutChannel_3: 3番目の畳み込み層の出力チャネル数
        @param Kernal_3: 3番目の畳み込み層のカーネルサイズ
        @returns: nn.Sequentialブロック
        """
        finalconv = nn.Sequential(
            nn.Conv2d(InChannel_1, OutChannel_1, Kernal_1, padding=0)
            , nn.PReLU()
            , nn.Conv2d(InChannel_2, OutChannel_2, Kernal_2, padding=1)
            , nn.PReLU()
            , nn.Conv2d(InChannel_3, OutChannel_3, Kernal_3, padding=0)
            , nn.PReLU()
        )
        return finalconv

    def __init__(self):
        """
        Nnetモデルを初期化する。
        エンコーダとデコーダの各ブロックを定義する。
        """
        super(Nnet, self).__init__()

        # エンコードパス
        self.conv_encode1 = self.ExtractionBlock(1, 8, 5, 8, 8, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv_encode2 = self.ExtractionBlock(8, 16, 3, 16, 16, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv_encode3 = self.ExtractionBlock(16, 32, 3, 32, 32, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv_encode4 = self.ExtractionBlock(32, 64, 3, 64, 64, 3)
        self.pool4 = nn.MaxPool2d(2)
        self.conv_encode5 = self.ExtractionBlock(64, 128, 3, 128, 128, 3)
        self.pool5 = nn.MaxPool2d(2)
        self.conv_encode6 = self.ExtractionBlock(128, 256, 3, 256, 256, 3)
        self.pool6 = nn.MaxPool2d(2)
        self.conv_encode7 = self.ExtractionBlock(256, 512, 3, 512, 512, 1)
        self.pool7 = nn.MaxPool2d(2)
        self.conv_encode8 = self.ExtractionBlock(512, 512, 1, 512, 512, 1)

        # デコードパス
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv1 = self.ExpansionBlock(1024, 512, 1, 512, 512, 3)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv2 = self.ExpansionBlock(512, 256, 3, 256, 256, 3)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv3 = self.ExpansionBlock(256, 128, 3, 128, 128, 3)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv4 = self.ExpansionBlock(128, 64, 3, 64, 64, 3)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv5 = self.ExpansionBlock(64, 32, 3, 32, 32, 3)

        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv6 = self.ExpansionBlock(32, 16, 3, 16, 16, 3)

        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Tconv7 = self.ExpansionBlock(16, 16, 5, 16, 16, 5)

        self.finalconv = self.FinalConv(16, 16, 1, 16, 8, 3, 8, 1, 1)

    def forward(self, image: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        Nnetモデルのフォワードパス。
        @param image: 入力画像テンソル
        @param prior: 先行画像テンソル
        @returns: 出力テンソル
        """
        # エンコードパス
        DownConv1 = self.conv_encode1(image)
        DownConv1_pool1 = self.pool1(DownConv1)

        DownConv2 = self.conv_encode2(DownConv1_pool1)
        DownConv2_pool2 = self.pool2(DownConv2)

        DownConv3 = self.conv_encode3(DownConv2_pool2)
        DownConv3_pool3 = self.pool3(DownConv3)

        DownConv4 = self.conv_encode4(DownConv3_pool3)
        DownConv4_pool4 = self.pool4(DownConv4)

        DownConv5 = self.conv_encode5(DownConv4_pool4)
        DownConv5_pool5 = self.pool5(DownConv5)

        DownConv6 = self.conv_encode6(DownConv5_pool5)
        DownConv6_pool6 = self.pool6(DownConv6)

        DownConv7 = self.conv_encode7(DownConv6_pool6)
        DownConv7_pool7 = self.pool7(DownConv7)

        DownConv8 = self.conv_encode8(DownConv7_pool7)

        # エンコードパス - 事前情報 (Prior)
        Prior_DownConv1 = self.conv_encode1(prior)
        PriorDownConv1_pool1 = self.pool1(Prior_DownConv1)

        Prior_DownConv2 = self.conv_encode2(PriorDownConv1_pool1)
        PriorDownConv2_pool2 = self.pool2(Prior_DownConv2)

        Prior_DownConv3 = self.conv_encode3(PriorDownConv2_pool2)
        PriorDownConv3_pool3 = self.pool3(Prior_DownConv3)

        Prior_DownConv4 = self.conv_encode4(PriorDownConv3_pool3)
        PriorDownConv4_pool4 = self.pool4(Prior_DownConv4)

        Prior_DownConv5 = self.conv_encode5(PriorDownConv4_pool4)
        PriorDownConv5_pool5 = self.pool5(Prior_DownConv5)

        Prior_DownConv6 = self.conv_encode6(PriorDownConv5_pool5)
        PriorDownConv6_pool6 = self.pool6(Prior_DownConv6)

        Prior_DownConv7 = self.conv_encode7(PriorDownConv6_pool6)
        PriorDownConv7_pool7 = self.pool7(Prior_DownConv7)

        Prior_DownConv8 = self.conv_encode8(PriorDownConv7_pool7)

        # デコードパス
        temp = torch.cat((Prior_DownConv8, DownConv8), dim=1)
        up1 = self.up1(temp)
        temp = torch.cat((Prior_DownConv7, DownConv7), dim=1)
        Tconv_1 = self.Tconv1(up1 + temp)

        up2 = self.up2(Tconv_1)
        temp = torch.cat((Prior_DownConv6, DownConv6), dim=1)
        Tconv_2 = self.Tconv2(up2 + temp)

        up3 = self.up3(Tconv_2)
        temp = torch.cat((Prior_DownConv5, DownConv5), dim=1)
        Tconv_3 = self.Tconv3(up3 + temp)

        up4 = self.up4(Tconv_3)
        temp = torch.cat((Prior_DownConv4, DownConv4), dim=1)
        Tconv_4 = self.Tconv4(up4 + temp)

        up5 = self.up5(Tconv_4)
        temp = torch.cat((Prior_DownConv3, DownConv3), dim=1)
        Tconv_5 = self.Tconv5(up5 + temp)

        up6 = self.up6(Tconv_5)
        temp = torch.cat((Prior_DownConv2, DownConv2), dim=1)
        Tconv_6 = self.Tconv6(up6 + temp)

        up7 = self.up7(Tconv_6)
        temp = torch.cat((Prior_DownConv1, DownConv1), dim=1)
        Tconv_7 = self.Tconv7(up7 + temp)

        out = self.finalconv(Tconv_7)
        return out
