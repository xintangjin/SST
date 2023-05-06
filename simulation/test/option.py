import argparse


parser = argparse.ArgumentParser(description="SST")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='1')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/SST_S_pale/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='SST_S', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default="/home/czy/NET/spectral/SST/simulation/train_code/exp/SST_S/2022_09_27_13_07_08/model/model_epoch_64.pth", help='pretrained model directory')


opt = parser.parse_args()


opt.mask_path = f"{opt.data_root}/TSA_simu_data/"
opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False