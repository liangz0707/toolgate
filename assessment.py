# coding:utf-8
__author__ = 'liangz14'
import numpy as np
import math

def psnr( img1,img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        mse = -1
    return 20 * math.log10(np.max(img1) / math.sqrt(mse))

def mse(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse
def rmse(img1,img2):
    rmse = np.sqrt(np.mean( (img1 - img2) ** 2 ))
    return rmse