import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from model.network import CRFBN
from datasets.datasets import CloudRemovalDataset, RICEDataset, WHUDataset
from loss.loss import currLoss
import time
from torchvision import transforms
from utils import to_psnr, to_ssim_skimage, validation, print_log
import os
import math
from PIL import Image
import cv2

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.0001')
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument('--net', default='', help="path to net_Dehazing (to continue training)")
parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[1], type=list)
opt = parser.parse_args()
print(opt)


# --- Learning rate decay strategy --- #
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

# ---  hyper-parameters for training and testing the neural network --- #
train_data_dir = './data/WHUS2_CR/train/'
val_data_dir = './data/WHUS2_CR/test/'
train_batch_size = opt.batchSize
val_batch_size = opt.testBatchSize
train_epoch = opt.nEpochs
data_threads = opt.threads
GPUs_list = opt.n_GPUs


device_ids = GPUs_list
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
print('===> Building model')
model = CRFBN(nums=6, iter_steps=3)


# --- Define the MSE loss --- #
L1_Loss = nn.L1Loss()
L1_Loss = L1_Loss.to(device)


# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)


# --- Build optimizer and scheduler --- #
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999)) 

# --- Load training data and validation/test data --- #
train_dataset = WHUDataset(root_dir=train_data_dir, transform=transforms.Compose([transforms.ToTensor()]))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=data_threads, shuffle=True)

val_dataset = WHUDataset(root_dir=val_data_dir, transform=transforms.Compose([transforms.ToTensor()]), train=False)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=data_threads, shuffle=False)


old_val_psnr, old_val_ssim = validation(model, val_dataloader, device, save_tag=False)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

for epoch in range(1, opt.nEpochs + 1):
    print("Training...")
    start_time = time.time()
    # learning rate decay
    lr=lr_schedule_cosdecay(epoch, opt.nEpochs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  
    psnr_list = []
    ssim_list = []
    for iteration, inputs in enumerate(train_dataloader,1):

        cloud, gt = Variable(inputs['cloud_image']), Variable(inputs['ref_image'])
        cloud = cloud.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        model.train()
        cloud_removal = model(cloud)

        Loss = currLoss(cloud_removal, gt, device)

        Loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print("===>Epoch[{}]({}/{}): Loss: {:.5f} lr: {:.6f} Time: {:.2}min ".format(epoch, iteration, len(train_dataloader), Loss.item(), lr, (time.time()-start_time)/60))
            
        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(cloud_removal[-1], gt))
        ssim_list.extend(to_ssim_skimage(cloud_removal[-1], gt))
    
    train_psnr = sum(psnr_list) / len(psnr_list)
    train_ssim = sum(ssim_list) / len(ssim_list)
    save_checkpoints = './checkpoints/WHUS2_CR/'
    if os.path.isdir(save_checkpoints)== False:
        os.mkdir(save_checkpoints)

    # --- Save the network  --- #
    torch.save(model.state_dict(), './checkpoints/WHUS2_CR/cloud_removal_{}.pth'.format(epoch))

    # --- Use the evaluation model in testing --- #
    model.eval()

    val_psnr, val_ssim = validation(model, val_dataloader, device, save_tag=False)
    
    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(model.state_dict(), './checkpoints/WHUS2_CR/cloud_removal_best.pth')
        old_val_psnr = val_psnr
        
    print_log(epoch, train_epoch, train_psnr, train_ssim, val_psnr, val_ssim, 'whus2-cr')
