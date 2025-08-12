## test XCAT female 512
import numpy as np
from PIL import Image
from NNet.TestingDataset_Nnet_XCAT import *
from NNet.model_Nnet import Nnet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_PATH = './experiments/Nnet/20250701_110121/trained_model/nnet_medical_ct_best_epoch50_loss0.0334.pth'
model = Nnet()
model.load_state_dict(torch.load(model_PATH, map_location='cuda:0'))
model.to(device)
model.eval()
## set the save path
root_save = r'./prediction/Nnet_XCATfemale/test/'
x_transform = T.ToTensor()
y_transform = T.ToTensor()
SliceNum = 160

test_dataset_4DCBCT = Test_XCATdataset(
    './test_dataset/'
    , './test_dataset/'
    , './test_dataset/'
    , SliceNum
    , transform=x_transform
    , target_transform=y_transform
)
test_dataloader = DataLoader(
    test_dataset_4DCBCT
    , batch_size=1
    , shuffle=False
    , num_workers=0
)

for i, batch in enumerate(test_dataloader, 0):
    # [N, 1, H, W]
    images, prior = batch
    images = images.to(device)
    prior = prior.to(device)
    PhaseIndex, SliceIndex = divmod(i, SliceNum)
    path = root_save + '/CNNPhase' + str(PhaseIndex + 1)
    os.makedirs(path, exist_ok=True)

    with torch.no_grad():
        outputs = model(images, prior)
    output = outputs[0][0].mul(255).cpu().detach().squeeze().numpy()
    output_normal = np.uint8(np.interp(output, (output.min(), output.max()), (0, 255)))
    im = Image.fromarray(output_normal)
    im.save(root_save + '/CNNPhase' + str(PhaseIndex + 1) + '/Processed' + str(SliceIndex + 1) + '.png')

print('Print the processed image by N-Net...')
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
