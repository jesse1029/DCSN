

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
from math import acos, degrees
from tensorboardX import SummaryWriter 
from trainOps import *


# torch.backends.cudnn.benchmark=True
# Hyperparameters
batch_size = 36
device = 'cuda'
MAX_EP = 16000
VAL_HR = 256
INTERVAL= 4
WIDTH=4
BANDS = 172
SIGMA = 0.0    ## Noise free -> SIGMA = 0.0
               ## Noise mode -> SIGMA > 0.0
SOURCE = '4fig'
TARGET = '4fig'
CR = 1        ## CR = 1, 5, 10, 15, 20
prefix='DCSN'
SNR = 25

    
def trainer():
    ## Reading files #
    
    ## Load test files from specified text with SOURCE/TARGET name (you can replace it with other path u want)
    flist = loadTxt('train_%s.txt' % SOURCE)
    valfn = loadTxt('val_%s.txt' % TARGET)
    tlen = len(flist)

    train_loader = torch.utils.data.DataLoader(dataset_h5(flist, width=4, marginal=60, root=''), batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = torch.utils.data.DataLoader(dataset_h5(valfn, mode='Validation',root=''), batch_size=5, shuffle=False, pin_memory=False)

    model = DCSN(snr=0, cr=CR)   ## cr =[1, 5, 10, 15, 20] compression ratio
    model = torch.nn.DataParallel(model).to(device)
    
    ### state_dict = torch.load('ckpt/DCSN_all_cr_1.pth')  ## finetune
    ### model.load_state_dict(state_dict)                  ## finetune
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)  ## finetune lr=0.00003
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    L1Loss = torch.nn.L1Loss()

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('Rec'):
        os.mkdir('Rec')    
    writer = SummaryWriter('log/%s_exp2_%s_%s_%s' % (prefix, SOURCE, TARGET, CR))
    
    resume_ind = 0
    step = resume_ind
    best_sam = 99
    for epoch in range(resume_ind, MAX_EP): 
        ep_loss = 0.
        for batch_idx, (x,_) in enumerate(train_loader):
            running_loss=0.
            
            optimizer.zero_grad()
            x = x.view(x.size()[0]*x.size()[1], x.size()[2], x.size()[3], x.size()[4])
            x = x.to(device).permute(0,3,1,2).float()
            decoded, _ = model(x)
            loss = L1Loss(decoded, x)
            loss.backward()
            
            optimizer.step()
            running_loss  +=  loss.item()
            
        if epoch% 10 ==0 :
            with torch.no_grad():
                rmses, sams, fnames, psnrs = [], [], [], []
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
                        rmses.append(rmse(GT, img[bt]))
                        fnames.append(save_path)
                        psnrs.append(psnr(GT, img[bt]))
                
                ep = time.time()-start_time
                ep = ep / len(sams)
                plog('[epoch: %d, batch: %5d] loss: %.3f, , val-RMSE: %.3f, val-SAM: %.3f, val-PSNR: %.3f, AVG-Time: %.3f' %
                      (epoch, batch_idx+resume_ind, running_loss, np.mean(rmses), np.mean(sams), np.mean(psnrs), ep)
                      , prefix, SOURCE, TARGET, CR)
                ## Dump the SAM/RMSE
                writer.add_scalar('Validation RMSE', np.mean(rmses), step)
                writer.add_scalar('Validation SAM', np.mean(sams), step)
                scheduler.step(np.mean(sams))
                
                with open('log/validataion_%s_%s_%d.txt' % (SOURCE, TARGET, CR),'w') as fp:
                    for r, s, f in zip(rmses, sams, fnames):
                        fp.write("%s:\tRMSE:%.4f\tSAM:%.3f\n" % (f, r, s))
                
            if best_sam > np.mean(sams):
                best_sam = np.mean(sams)
                torch.save(model.state_dict(), 'checkpoint/%s_%s_%s_cr_%d_epoch_%d_%.3f.pth' %(prefix, SOURCE, TARGET, CR, epoch, np.mean(sams)))
                
            ep_loss += running_loss
            writer.add_scalar('Running loss', running_loss, step)
                
            running_loss = 0.0
            running_loss2 = 0.0
            model.train() 
        
                
        if epoch% 50 ==0 and epoch > 1:
            torch.save(model.state_dict(), 'checkpoint/%s_%s_%s_cr_%d_epoch_%d.pth' %(prefix, SOURCE, TARGET, CR, epoch))
        
        step+=1


if __name__ == '__main__':
    trainer()
