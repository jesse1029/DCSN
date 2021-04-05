
import os
import glob
import time
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import torch.nn as nn
import json
from sklearn.metrics import log_loss
import pdb
import random as rn
from model import *
from utils import *
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from dataset import *
from scipy.io import savemat, loadmat
import sewar
from math import acos, degrees
from tensorboardX import SummaryWriter 
from trainOps import *


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# torch.backends.cudnn.benchmark=True
# Hyperparameters
batch_size = 32
device = 'cuda' ## cpu or cuda (set cuda if gpu avaiilable)
VAL_HR = 256
INTERVAL= 4
WIDTH=4
BANDS = 172
CR = 1        ## CR = 1, 5, 10, 15, 20
SIGMA = 0.0   ## Noise free -> SIGMA = 0.0
              ## Noise mode -> SIGMA > 0.0
TARGET = '4fig'
SNR = 99999

 

if not os.path.isdir('Rec'):
    os.mkdir('Rec')

## Load test files from specified text with TARGET name (you can replace it with other path u want)
valfn = loadTxt('val_%s.txt' % TARGET)  

## Setup the dataloader
val_loader = torch.utils.data.DataLoader(dataset_h5(valfn, mode='Validation',root=''), batch_size=5, shuffle=False, pin_memory=False)

model = DCSN(cr=CR)  
state_dict = torch.load('ckpt/DCSN_all_cr_1.pth' ,map_location=device)

if device=='cpu':
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
else:
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(state_dict)
    
model.eval()
with torch.no_grad():
    rmses, sams, fnames, psnrs = [], [], [],[]
    start_time = time.time()
    for ind2, (vx, vfn) in enumerate(val_loader):
        model.eval()
        vx = vx.view(vx.size()[0]*vx.size()[1], vx.size()[2], vx.size()[3], vx.size()[4])
        vx= vx.to(device).permute(0,3,1,2).float()
        if SIGMA>0:
            val_dec = model(awgn(model(vx, mode=1), SNR), mode=2)
            
        else:
            val_dec,_ = model(vx)


        ## Recovery to image HSI
        val_batch_size = len(vfn)
        img = [np.zeros((VAL_HR, VAL_HR, BANDS)) for _ in range(val_batch_size)]
        val_dec = val_dec.permute(0,2,3,1).cpu().numpy()
        cnt = 0

        for bt in range(val_batch_size):
            for z in range(0, VAL_HR, INTERVAL):
                img[bt][:,z:z+WIDTH,:] = val_dec[cnt]
                cnt +=1
            save_path = vfn[bt].split('/')
            save_path = save_path[-2] + '-' + save_path[-1]
            np.save('Rec/%s.npy' % (save_path), img[bt])

            GT = lmat(vfn[bt]).astype(np.float)
            maxv, minv=np.max(GT), np.min(GT)
            img[bt] = img[bt]*(maxv-minv) + minv ## De-normalization
            sams.append(sam(GT, img[bt]))
            psnrs.append(psnr(GT, img[bt]))
            rmses.append(np.sqrt(np.mean((GT-img[bt])**2)))
            fnames.append(save_path)
            print('{:25} '.format(vfn[bt].split('/')[-1])+' PSNR: %.3f RMSE: %.3f SAM: %.3f' %
            ( psnrs[bt], rmses[bt], sams[bt]))
    
    ep = time.time()-start_time
    ep = ep / len(sams)
    print('\n\nval-RMSE: %.3f, val-SAM: %.3f, psnr:%.3f, AVG-Time: %.3f , CR: %.f, SNR: %.f' %
          (np.mean(rmses), np.mean(sams), np.mean(psnrs), ep, CR, SNR))
    print('%.3f / %.3f / %.3f' %
          (np.mean(psnrs), np.mean(rmses), np.mean(sams)))

