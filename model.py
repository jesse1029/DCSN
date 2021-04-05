import torch
from torch import nn
import numpy as np, math
from torch.nn import functional as F
from torch.autograd import Variable
import functools
from module_util import *

import pdb



##===============MFB + MFA ===============================================
class MultiScaleFeatFusionBlock(nn.Module):   ## MFB
    def __init__(self, nf=64, gc=32, bias=True):
        super(MultiScaleFeatFusionBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class MultiScaleFeatAggregation(nn.Module):   ## MFA


    def __init__(self, nf, gc=32):
        super(MultiScaleFeatAggregation, self).__init__()
        self.MFB1 = MultiScaleFeatFusionBlock(nf, gc)
        self.MFB2 = MultiScaleFeatFusionBlock(nf, gc)
        self.MFB3 = MultiScaleFeatFusionBlock(nf, gc)

    def forward(self, x):
        out = self.MFB1(x)
        out = self.MFB2(out)
        out = out * 0.2 + x
        out = self.MFB3(out)
        return out * 0.2 + x


class DCSNDec(nn.Module):
    def make_layer(block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def __init__(self, in_nc, out_nc, nf, nb, gc=32, up_scale = 4):
        super(DCSNDec, self).__init__()
        MFA_block_f = functools.partial(MultiScaleFeatAggregation, nf=nf, gc=gc)
        self.up_scale = up_scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.MFA_trunk = make_layer(MFA_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 1, 1, 0, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.MFA_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.up_scale == 4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea,  scale_factor=2, mode='nearest')))
        fea = self.conv_last(self.lrelu(self.HRconv(fea)))
    

        return fea
        
##=========DCSN full module==========
class DCSN(nn.Module): 
    def __init__(self, snr=0, cr=1):
        super(DCSN, self).__init__()
        self.snr = snr
        if cr == 1:
            last_stride = 2
            last_ch = 27
            last_kernel_w = 1
            last_padding_w = 0
        else:
            last_stride = 1 
            last_kernel_w = 2
            last_padding_w = 1
            
        up_scale = 4 if cr<5 else 2
        if cr==5:
            last_ch = 32
        elif cr==10:
            last_ch = 64
        elif cr==15:
            last_ch=103
        elif cr==20:
            last_ch=140
        
        self.encoder = nn.Sequential(
            nn.Conv2d(172, 128, [3, 3], stride=[2,2], padding=[1,0]),  # b, 16, 10, 10
            nn.LeakyReLU(True),
            nn.Conv2d(128, 64, [3,1], stride=[1,1], padding=[1,0]),  # b, 8, 3, 3
            nn.LeakyReLU(True),
            nn.Conv2d(64, last_ch, [3,last_kernel_w], stride=[last_stride, 1], padding=[1, last_padding_w])
        )
        print(self.encoder)
        
        ##  128*4*172=88064 --> 32*1*27 --> cr=1%
        ##
        ## 64*2*64 --> cr=9.30%
        ## 64*2*32 --> cr=4.65%
        ## 64*2*103--->cr=14.97%
        ## 64*2*140 -->cr=20.3%
        
        self.decoder = DCSNDec(last_ch, 172, 64, 16, up_scale=up_scale)
       
    def awgn(self, x, snr):
        snr = 10**(snr/10.0)
        xpower = torch.sum(x**2)/x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape).cuda() * npower


    def forward(self, data, mode=0): ### Mode=0, default, mode=1: encode only, mode=2: decoded only
        
        if mode==0:
            x = self.encoder(data)
            if self.snr > 0:
                x = self.awgn(x, self.snr)
            y = self.decoder(x)
            return y, x
        elif mode==1:
            return self.encoder(data)
        elif mode==2:
            return self.decoder(data)
        else:
            return self.decoder(self.encoder(data))
