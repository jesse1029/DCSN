import numpy as np
from scipy.io import savemat, loadmat
import torch
import math

def plog(msg, prefix, SOURCE,TARGET, CR):
    print(msg)
    with open('log/%s_mylog_%s_%s_%d.txt' % (prefix, SOURCE, TARGET, CR), 'a') as fp:
        fp.write(msg+"\n")
        
def sam(x, y):
    '''
    num = sum(x .* y, 3);
den = sqrt(sum(x.^2, 3) .* sum(y.^2, 3));
sam = sum(sum(acosd(num ./ den)))/(n_samples);
    '''
    num = np.sum(np.multiply(x, y), 2)
    den = np.sqrt(np.multiply(np.sum(x**2, 2), np.sum(y**2, 2)))
    sam = np.sum(np.degrees(np.arccos(num / den))) / (x.shape[0]*x.shape[1])
    return sam

def psnr(x,y):
    bands = x.shape[2]
    x = np.reshape(x, [-1, bands])
    y = np.reshape(y, [-1, bands])
    msr = np.mean((x-y)**2, 0)
    maxval = np.max(y, 0)**2
    return np.mean(10*np.log10(maxval/msr))


def lmat(fn):
    x=loadmat(fn)
    x=x[list(x.keys())[-1]]
   
    return x
        
def loadTxt(fn):
    a = []
    with open(fn, 'r') as fp:
        data = fp.readlines()
        for item in data:
            fn = item.strip('\n')
            a.append(fn)
    return a

def rmse(x, y):
    aux = np.sum(np.sum((x-y)**2, 0),0) / (x.shape[0]*x.shape[1])
    rmse_per_band = np.sqrt(aux)
    rmse_total = np.sqrt(np.sum(aux) / x.shape[2])
    return rmse_total

def awgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/x.numel()
    npower = torch.sqrt(xpower / snr)
    return x + torch.randn(x.shape).cuda() * npower