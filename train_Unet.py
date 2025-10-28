import torch, os
from monai.networks.nets import FlexibleUNet, SwinUNETR

bundle_dir = "/media/zzg/GJ_disk01/pretrained_model/MONAI/spleen_ct_segmentation"
model_file = os.path.join(bundle_dir, "models", "model.pt")
sd = torch.load(model_file, map_location='cpu')  # sd 可能直接是 state_dict 或 dict 包含 state_dict

# 构造与 bundle 相同的模型架构（非常重要：架构要匹配）
model = FlexibleUNet(in_channels=1,
                     out_channels=1,
                     backbone='efficientnet-b0',
                     pretrained=False,
                     spatial_dims=2)

# 如果 sd 是完整 state_dict，直接加载（strict=False 更容错）
state_dict = sd if 'state_dict' not in sd else sd['state_dict']
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print('missing keys:', missing, 'unexpected keys:', unexpected)
