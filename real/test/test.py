import torch
import os
import argparse
from utils import dataparallel
import scipy.io as sio
import numpy as np
from torch.autograd import Variable


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
parser.add_argument('--data_path', default='TO your TSA_real_data Measurements path', type=str,help='path of data')
parser.add_argument('--mask_path', default='TO your TSA_real_data mask path', type=str,help='path of mask')
parser.add_argument("--size", default=512, type=int, help='the size of trainset image')
parser.add_argument("--trainset_num", default=2000, type=int, help='total number of trainset')
parser.add_argument("--testset_num", default=5, type=int, help='total number of testset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
parser.add_argument("--pretrained_model_path", default=None, type=str)
opt = parser.parse_args()
print(opt)


def prepare_data(path, file_num=5):
    HR_HSI = np.zeros((660,714))
    path1 = os.path.join(path) + 'scene1.mat'
    data = sio.loadmat(path1)
    HR_HSI[:,:] = data['meas_real']
    HR_HSI[HR_HSI < 0] = 0.0
    HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI

def load_mask(path,size=660):
    ## load mask
    data = sio.loadmat(path)
    mask = data['mask']
    mask_3d = np.tile(mask[np.newaxis, :, :, np.newaxis], (1, 1, 1, 28))
    mask_3d_shift = np.zeros((size, size + (28 - 1) * 2, 28))
    mask_3d_shift[:, 0:size, :] = mask_3d
    for t in range(28):
        mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
    mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
    mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
    mask_3d = mask_3d.transpose(0, 3, 1, 2)
    mask_3d = torch.FloatTensor(mask_3d.copy())
    return mask_3d, mask_3d_shift.unsqueeze(0), mask_3d_shift_s.unsqueeze(0)

HR_HSI = prepare_data(opt.data_path, 5)
HR_HSI = HR_HSI[24:536, 96:662]
Mask, mask_3d_shift, mask_3d_shift_s = load_mask('to your mask.mat path')
Mask = Mask[:,:, 24:536, 96:608]
Mask = Variable(Mask)
Mask = Mask.cuda()
save_path = './exp/sst/'
pretrained_model_path = "to your pretrained model path"
model = torch.load(pretrained_model_path)



model = model.eval()
model = dataparallel(model, 1)
psnr_total = 0
k = 0
with torch.no_grad():

    meas = HR_HSI
    meas = meas / meas.max() * 0.8
    meas = torch.FloatTensor(meas)

    input = meas.unsqueeze(0)
    input = Variable(input)
    input = input.cuda()
    mask_3d_shift = mask_3d_shift.cuda()
    mask_3d_shift_s = mask_3d_shift_s.cuda()

    out = model(input, mask=Mask)
    result = out
    result = result.clamp(min=0., max=1.)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
    save_file = save_path + 'sst.mat'
    sio.savemat(save_file, {'res':res})
