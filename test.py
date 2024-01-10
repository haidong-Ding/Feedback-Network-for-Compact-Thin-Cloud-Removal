from __future__ import print_function  
import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from model.network_1 import CRFBN
from model.rcanet import CANet
from datasets.datasets import CloudRemovalDataset, RICEDataset, WHUDataset, MultiSpectralDataset
from os.path import exists, join, basename
import time
from torchvision import transforms
from utils import to_psnr, validation
import os
from thop import profile

# ---  hyper-parameters for testing the neural network --- #
test_data_dir = './data/Multi_spectral/'
test_batch_size = 1
data_threads = 15


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
test_dataset = MultiSpectralDataset(root_dir = test_data_dir, crop=False, transform = transforms.Compose([transforms.ToTensor()]), train=False)
test_dataloader = DataLoader(test_dataset, batch_size = test_batch_size, num_workers=data_threads, shuffle=False)

# --- Define the network --- #
# model = CRFBN(nums=6, iter_steps=3)
model = CANet()

# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=[0])


# --- Load the network weight --- #
model.load_state_dict(torch.load('./checkpoints/RCA_Net/Multi_spectral/cloud_removal_100.pth'))

# if isinstance(model,torch.nn.DataParallel):
#      model = model.module
   
# --- Use the evaluation model in testing --- #
model.eval()
print('--- Testing starts! ---')
print(len(test_dataloader))

total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))

test_psnr, test_ssim, avg_time = validation(model, test_dataloader, device, save_tag=True, save_path='./results/RCA-Multi-Spectral/')
print('test_psnr: {0:.6f}, test_ssim: {1:.6f}, avg_time: {2:.6f}'.format(test_psnr, test_ssim, avg_time))
