import os
from option import opt
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
from architecture import *
from utils import *
import scipy.io as scio
import torch
import time
import numpy as np


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# Intialize mask
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path,  10)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def test(model):
    psnr_list, ssim_list = [], []
    test_data = LoadTest(opt.test_path)
    # test_data = test_data[4,:,:,:]
    # test_data = test_data.unsqueeze(0)
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch_test)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        model_out = model(input_meas, input_mask_test, mask3d_batch_test)
    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    print('===> testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(psnr_mean, ssim_mean, (end - begin)))
    model.train()
    return pred, truth


def main():
    # model
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()
    pred, truth = test(model)
    name = opt.outf + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})

if __name__ == '__main__':
    main()