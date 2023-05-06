import argparse


parser = argparse.ArgumentParser(description="SST_S")

# Hardware specifications
parser.add_argument("--USE_MULTI_GPU", type=bool, default=False)
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='to your datasets', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/SST_S/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='SST_S', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')

# Training specifications
parser.add_argument('--batch_size', type=int, default=5, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', help='You can set various templates in option.py')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)

opt = parser.parse_args()

# dataset
opt.data_path = f"{opt.data_root}/cave_1024_28/"
opt.mask_path = f"{opt.data_root}/TSA_simu_data/"
opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False