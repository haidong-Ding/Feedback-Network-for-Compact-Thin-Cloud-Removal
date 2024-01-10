import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.calayer = nn.Sequential(nn.Conv2d(channels, channels//8, 1, padding=0, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(channels//8, channels, 1, padding=0, bias=True),
                                     nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.calayer(y)
        return x*y


class PixelAttention(nn.Module):
    def __init__(self, channels):
        super(PixelAttention, self).__init__()
        self.palayer = nn.Sequential(nn.Conv2d(channels, channels//8, 1, padding=0, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(channels//8, 1, 1, padding=0, bias=True),
                                     nn.Sigmoid())

    def forward(self, x):
        y = self.palayer(x)
        return x*y


class BasicBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=kernel_size,
                               padding=(kernel_size//2),
                               bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=dim*2,
                               out_channels=dim,
                               kernel_size=kernel_size,
                               padding=(kernel_size//2),
                               bias=True)

        # Attention
        self.ca = ChannelAttention(dim)
        self.pa = PixelAttention(dim)


    def forward(self, x):
        x0 = x
        x = self.act(self.conv1(x))
        x = torch.cat((x, x0), dim=1)
        x = self.conv2(x)
        x = self.ca(x)
        x = self.pa(x)

        return x + x0


class FeedbackBlock(nn.Module):
    def __init__(self, dim, kernel_size, nums):
        super(FeedbackBlock, self).__init__()
        layers = [BasicBlock(dim, kernel_size) for _ in range(nums)]

        self.fb = nn.Sequential(*layers)

    def forward(self, x):
        res = self.fb(x)

        return res + x


class FeatureFusion(nn.Module):
    def __init__(self, dim, kernel_size):
        super(FeatureFusion, self).__init__()
        # feature extraction
        self.fe = nn.Sequential(nn.Conv2d(3, dim, kernel_size, padding=(kernel_size//2), bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(dim, dim, 1, padding=0, bias=True))

        # multi-scale feature fusion
        self.conv_scale0 = nn.Conv2d(dim, dim, 1, padding=0, bias=True)
        self.conv_scale1 = nn.Conv2d(dim, dim, 1, padding=0, bias=True)
        self.conv_scale2 = nn.Conv2d(dim, dim, 1, padding=0, bias=True)

        self.post_conv = nn.Conv2d(dim*4, dim, 3, padding=1, bias=True)

    def forward(self, input, feedback):
        _, _, H, W = input.size()
        # extracting features
        f_i = self.fe(input)
        f_f = self.fe(feedback)

        # feaure fusion
        f = f_i + f_f
        f_scale0 = self.conv_scale0(f)
        
        f_scale0 = F.interpolate(f, (H//2, W//2), mode='bilinear', align_corners=False)
        f_scale0 = self.conv_scale0(f_scale0)
        f_scale0 = F.interpolate(f_scale0, (H, W), mode='bilinear', align_corners=False)

        f_scale1 = F.interpolate(f, (H//4, W//4), mode='bilinear', align_corners=False)
        f_scale1 = self.conv_scale1(f_scale1)
        f_scale1 = F.interpolate(f_scale1, (H, W), mode='bilinear', align_corners=False)

        f_scale2 = F.interpolate(f, (H//8, W//8), mode='bilinear', align_corners=False)
        f_scale2 = self.conv_scale2(f_scale2)
        f_scale2 = F.interpolate(f_scale2, (H, W), mode='bilinear', align_corners=False)

        f = torch.cat((f, f_scale0, f_scale1, f_scale2), dim=1)

        f = self.post_conv(f)

        return f


class CRFBN(nn.Module):
    def __init__(self, dim=32, kernel_size=3, nums=6, iter_steps=3):
        super(CRFBN, self).__init__()
        self.ff = FeatureFusion(dim=dim, kernel_size=kernel_size)  # ff: feature fusion
        self.fb = FeedbackBlock(dim=dim, kernel_size=kernel_size, nums=nums)  # fb: feedback

        self.post = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=True),
                                  nn.Conv2d(dim, 3, 3, padding=1, bias=True))
        self.steps = iter_steps

    def forward(self, x):
        outputs = []
        feedback = x
        for step in range(self.steps):
            f = self.ff(x, feedback)
            f = self.fb(f)
            out = self.post(f)
            feedback = out
            outputs.append(out)

        return outputs
