import torch
from .SST_S import SST_S
from .SST_M import SST_M
from .SST_L import SST_L
from .SST_LPlus import SST_LPlus

def model_generator(method, pretrained_model_path=None):
    if method == 'SST_S':
        model = SST_S(in_channels=28, out_channels=28, n_feat=28, stage=1).cuda()
    elif method == 'SST_M':
        model = SST_M(in_channels=28, out_channels=28, n_feat=28, stage=1).cuda()
    elif method == 'SST_L':
        model = SST_L(in_channels=28, out_channels=28, n_feat=28, stage=1).cuda()
    elif method == 'SST_LPlus':
        model = SST_LPlus(in_channels=28, out_channels=28, n_feat=28, stage=1).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
    return model