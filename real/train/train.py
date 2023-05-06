from architecture import *
from utils import *
from dataset import dataset
import torch.utils.data as tud
import torch
import torch.nn.functional as F
import time
import datetime
from torch.autograd import Variable
import os
from option import opt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# load training data
CAVE = prepare_data_cave(opt.data_path_CAVE, 30)
KAIST = prepare_data_KAIST(opt.data_path_KAIST, 30)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = os.path.join(opt.outf, date_time)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# model
model = model_generator(opt.method, opt.pretrained_model_path).cuda()


# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
criterion = nn.L1Loss()

if __name__ == "__main__":

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    ## pipline of training
    for epoch in range(1, opt.max_epoch):
        model.train()
        Dataset = dataset(opt, CAVE, KAIST)
        loader_train = tud.DataLoader(Dataset, num_workers=8, batch_size=opt.batch_size, shuffle=True)

        epoch_loss = 0

        start_time = time.time()
        for i, (input, label, Mask, Phi, Phi_s) in enumerate(loader_train):
            input, label, Mask, Phi, Phi_s = Variable(input), Variable(label), Variable(Mask), Variable(Phi), Variable(
                Phi_s)
            input, label, Mask, Phi, Phi_s = input.cuda(), label.cuda(), Mask.cuda(), Phi.cuda(), Phi_s.cuda()
            Mask = Mask.permute(0, 3, 1, 2)
            input_mask = init_mask(Mask, Phi, Phi_s, opt.input_mask)

            # out = model(input, input_mask)
            out = model(input, mask=Mask)
            loss = criterion(out, label)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if i % (1000) == 0:
                print('%4d %4d / %4d loss = %.10f time = %s' % (
                    epoch + 1, i, len(Dataset) // opt.batch_size, epoch_loss / ((i + 1) * opt.batch_size),
                    datetime.datetime.now()))
        scheduler.step()
        elapsed_time = time.time() - start_time
        print('epcoh = %4d , loss = %.10f , time = %4.2f s, lr = %.10f' % (
        epoch + 1, epoch_loss / len(Dataset), elapsed_time, optimizer.param_groups[0]["lr"]))
        torch.save(model, os.path.join(opt.outf, 'model_%03d.pth' % (epoch + 1)))
