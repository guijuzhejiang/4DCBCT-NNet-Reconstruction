## XCAT female 512のテスト
import numpy as np
from PIL import Image
from test_dataset_Nnet import Test_Dataset
from model_Nnet import Nnet
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from config import DATASET_CONFIG


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_PATH = './experiments/Nnet/20250813_140957/trained_model/nnet_medical_ct_best_epoch3_loss0.037521.pth'
model = Nnet()
model.load_state_dict(torch.load(model_PATH, map_location='cuda:0'))
model.to(device)
model.eval()
## 保存パスを設定
root_save = r'./prediction/Nnet/'


test_dataset = Test_Dataset(DATASET_CONFIG['data_root'], DATASET_CONFIG['test_dataset_indices'])
test_dataloader = DataLoader(
    test_dataset
    , batch_size=1
    , shuffle=False
    , num_workers=0
)

for i, batch in enumerate(test_dataloader, 0):
    # [N, 1, H, W]
    images, prior, img_path = batch
    images = images.to(device)
    prior = prior.to(device)
    dir_list = img_path[0].split('/')
    save_path = os.path.join(root_save, dir_list[-5], dir_list[-4], dir_list[-3])
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        outputs = model(images, prior)
    output = outputs[0][0].mul(255).cpu().detach().squeeze().numpy()
    output_normal = np.uint8(np.interp(output, (output.min(), output.max()), (0, 255)))
    im = Image.fromarray(output_normal)
    png_dir = os.path.join(save_path, dir_list[-1].split('.')[0] + '.png')
    im.save(png_dir)

print('N-Netによる処理済み画像を出力します...')
print(outputs[0][0][200][300])
plt.subplot(1, 3, 1)
plt.imshow(images[0][0].cpu().detach().numpy())
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(prior[0][0].cpu().detach().numpy())
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(outputs[0][0].cpu().detach().numpy())
plt.axis('off')
