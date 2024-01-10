import torch.nn as nn
import torchvision.transforms as transforms
import math


def currLoss(outputs, ref, device, kernel_size=3):
    T = len(outputs) # number of iterations
    L1 = nn.L1Loss().to(device) # loss function
    # Gaussian blur
    if T==1:
        loss = L1(outputs[-1], ref)
    elif T==2:
        sigma = math.cos(math.pi*1/(2*T))
        transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        target = transform(ref)
        loss = L1(outputs[0], target)
        loss += L1(outputs[-1], ref)
    else:
        sigma = math.cos(math.pi*1/(2*T))
        transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        target = transform(ref)
        loss = L1(transform(outputs[0]), target)
        for i in range(1, len(outputs)-1):
            sigma = math.cos(math.pi * (i+1) / (2 * T))
            transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            target = transform(ref)
            loss += L1(transform(outputs[i]), target)

        loss += L1(outputs[-1], ref)

    return loss
    
