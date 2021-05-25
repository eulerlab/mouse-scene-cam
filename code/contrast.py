# -*- coding: utf-8 -*-
"""
@author: Yongrong Qiu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import cv2
import glob
import h5py
from skimage.morphology import disk
from scipy.stats import pearsonr
from scipy.stats import skew
from scipy.ndimage import gaussian_filter

#analysis unit: could be one region, region list, one frame or frame list
#frame_num: list of frame numbers to be used
#frame region to be used: [x1:x2,y1:y2] (x1 x2 in height direction, y1 y2 in width direction)
#
def analysis_units_fun(frame_num,x1,x2,y1,y2):
    return movie_bgr_h5[frame_num,x1:x2,y1:y2,:]

#import data
#read movie real data, real means: after spectral calibration, before gamma correction for the screen
def read_movie_from_h5(filename):
    h5f = h5py.File(filename,'r')
    movie_bgr_h5=h5f['movie_bgr_real'][:]
    h5f.close()
    return movie_bgr_h5

#to better visulaize image, use gamma correction to transfer image real to image view
def img_real2view(img):
    gamma_correction=lambda x:np.power(x,1.0/2.2)
    img_shape=img.shape
    # gray image
    if np.size(img_shape)==2:
        #uint8
        if np.max(img)>1:
            temp_view=np.zeros_like(img,dtype=np.float32)
            temp_view=np.float32(img)/255.0#float32, 1.0
            temp_view=gamma_correction(temp_view)
            temp_view2=np.zeros_like(img,dtype=np.uint8)
            temp_view2=np.uint8(temp_view*255)
            return temp_view2
        #float
        if np.max(img)<2:
            return gamma_correction(img)
    #color image
    if np.size(img_shape)==3:
        #uint8
        if np.max(img)>1:
            temp_view=np.zeros_like(img,dtype=np.float32)
            temp_view=np.float32(img)/255.0#1.0
            temp_view=gamma_correction(temp_view)
            temp_view2=np.zeros_like(img,dtype=np.uint8)
            temp_view2=np.uint8(temp_view*255)#255
            return temp_view2
        #float
        if np.max(img)<2:
            return gamma_correction(img)

#function: gaussian kernel 1d
#input: sigma: std
#       order: A positive order corresponds to convolution with
#              that derivative of a Gaussian, use 0 here
#       radius: radius of the filter
def my_gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    p = np.polynomial.Polynomial([0, 0, -0.5 / (sigma * sigma)])
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(p(x), dtype=np.double)
    phi_x /= phi_x.sum()
    if order > 0:
        q = np.polynomial.Polynomial([1])
        p_deriv = p.deriv()
        for _ in range(order):
            # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
            # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
            q = q.deriv() + q * p_deriv
        phi_x *= q(x)
    return phi_x

#function: gaussian filter 2d
def my_gaussian_kernel2d(sigma,order,radius):
    g_ker_1d=my_gaussian_kernel1d(sigma, order, radius)
    g_ker_2d=np.outer(g_ker_1d, g_ker_1d)
    g_ker_2d /=g_ker_2d.sum()
    return g_ker_2d

#function: my difference of gaussian kernel 1d
#input: centersigma is the center sigma, surround sigma=1.5*centersigma, centersigma=RFradius
#       radius: defalt 3*centersigma 
#output: kernel size length: 1+3*centersigma*2
def my_DOG_kernel1d(centersigma,order,radius):
    surroundsigma=1.5*centersigma
    center_kernel1d=my_gaussian_kernel1d(centersigma,order,radius)
    surround_kernel1d=my_gaussian_kernel1d(surroundsigma,order,radius)
    out_kernel1d=center_kernel1d-surround_kernel1d
    return out_kernel1d

#function: my difference of gaussian kernel 2d, mimic retina center-surround onoff
#input: centersigma is the center sigma, surround sigma=1.5*centersigma
#       radius: kernelradius, defalt 3*centersigma 
#output: kernel size length: 1+3*centersigma*2
def my_DOG_kernel2d(centersigma,order,radius):
    surroundsigma=1.5*centersigma
    center_kernel2d=my_gaussian_kernel2d(centersigma,order,radius)
    surround_kernel2d=my_gaussian_kernel2d(surroundsigma,order,radius)
    out_kernel2d=center_kernel2d-surround_kernel2d
    return out_kernel2d

#function, calculate ONOFF for single pixel
#input:
#img: gray image, float, 1.0 (when phase srambled image, may be a little larger than 1.0)
#(xx,yy): center coordinate, xx: along height, yy: along width, RFradius: radius of center
#output:
#onoff value
def ONOFF_single(img,xx,yy,centersigma):
    surroundsigma=np.round(1.5*centersigma)
    kernelradius=3*centersigma
    temp=img[xx-kernelradius:xx+kernelradius+1,yy-kernelradius:yy+kernelradius+1]
    center_kernel2d=my_gaussian_kernel2d(centersigma,0,kernelradius)
    surround_kernel2d=my_gaussian_kernel2d(surroundsigma,0,kernelradius)
    centermean=np.sum(temp*center_kernel2d)
    surroundmean=np.sum(temp*surround_kernel2d)
    onoff=(centermean-surroundmean)/(centermean+surroundmean+1e-8)
    return onoff

#input: 
#centersigma is the center sigma
#img: image or image region, float
#output: onoff_img, float, -1.0 to 1.0
def onoff_wholeimg(img,centersigma):
    kernelradius=3*centersigma
    onoff_img=np.zeros((img.shape[0],img.shape[1]))
    for ii in np.arange(kernelradius,img.shape[0]-kernelradius-1):
        for jj in np.arange(kernelradius,img.shape[1]-kernelradius-1):
            onoff_img[ii,jj]=ONOFF_single(img,ii,jj,centersigma)
    if img.shape[0]==437:
        mask_con=np.zeros((437,437),np.uint8)
        cv2.circle(mask_con,(218,218),radius=218-kernelradius,color=255,thickness=-1)
        mask_con=np.float32(mask_con/255.0)
        onoff_img=np.multiply(onoff_img,mask_con)
    return onoff_img

#input: onoff_seed: random seed for contrast calculation
#onoff_num: random pick numbers
#centersigma is the center sigma
#img: image or image region, float 1.0 (when phase srambled, may be a little larger than 1.0)
#output: the onoff value distribution
def onoff_random(onoff_seed,onoff_num,centersigma,img):
    kernelradius=3*centersigma
    np.random.seed(onoff_seed+866)
    walk_height=np.random.choice(np.arange(kernelradius,img.shape[0]-kernelradius-1),onoff_num,replace=False)
    np.random.seed(onoff_seed+899)
    walk_width=np.random.choice(np.arange(kernelradius,img.shape[1]-kernelradius-1),onoff_num,replace=False)
    onoffs=np.zeros(onoff_num)
    for ii in range(onoff_num):
        onoffs[ii]=ONOFF_single(img,walk_height[ii],walk_width[ii],centersigma)
    return onoffs

#input: onoff_seed: random seed for contrast calculation
#onoff_num: total random pick numbers=numberofimages* each_random_pick_numbers
#centersigma is the center sigma
#imgs: images, all gray images, float 1.0 (when phase srambled, may be a little larger than 1.0)
#      format like: numberofimages*height*width
#output: the onoff value distribution
def onoff_random_imgs(onoff_seed,onoff_num,centersigma,imgs):
    num_imgs=imgs.shape[0]
    onoffs=[]
    for ii in range(num_imgs):
        onoffs.append(onoff_random(onoff_seed+ii,int(np.round(onoff_num/num_imgs)),centersigma,imgs[ii]))
    onoffs=np.array(onoffs)
    onoffs=onoffs.flatten()
    return onoffs

#input: onoff_seed: random seed for onoff and local contrast(rms2) calculation
#onoff_num: random pick numbers
#centersigma is the center sigma for onoff
#RFradius for local contrast(rms2)
#img: image or image region, float 1.0 (when phase srambled, may be a little larger than 1.0)
#output: the onoff and local contrast (rms2) value distribution
def onoff_rms2_random(onoff_seed,onoff_num,centersigma,RFradius,img):
    kernelradius=3*centersigma
    np.random.seed(onoff_seed+1866)
    walk_height=np.random.choice(np.arange(kernelradius,img.shape[0]-kernelradius-1),onoff_num,replace=False)
    np.random.seed(onoff_seed+2899)
    walk_width=np.random.choice(np.arange(kernelradius,img.shape[1]-kernelradius-1),onoff_num,replace=False)
    onoffs=np.zeros(onoff_num)
    rms2s=np.zeros(onoff_num)
    tempdisk=np.float64(disk(RFradius))
    for ii in range(onoff_num):
        onoffs[ii]=ONOFF_single(img,walk_height[ii],walk_width[ii],centersigma)
        temp=img[walk_height[ii]-RFradius:walk_height[ii]+RFradius+1,\
                 walk_width[ii]-RFradius:walk_width[ii]+RFradius+1]
        temp=temp[np.nonzero(tempdisk)]
        rms2s[ii]=np.std(temp,ddof=1)/(np.mean(temp)+1e-8)
    return onoffs,rms2s

#input: onoff_seed: random seed for contrast calculation
#onoff_num: total random pick numbers=numberofimages* each_random_pick_numbers
#centersigma is the center sigma for onoff
#RFradius for local contrast(rms2)
#imgs: images, all gray images, float 1.0 (when phase srambled, may be a little larger than 1.0)
#      format like: numberofimages*height*width
#output: the onoff and local contrast (rms2) value distribution
def onoff_rms2_random_imgs(onoff_seed,onoff_num,centersigma,RFradius,imgs):
    num_imgs=imgs.shape[0]
    onoffs=[]
    rms2s=[]
    for ii in range(num_imgs):
        temp_onoff,temp_rms2=onoff_rms2_random(onoff_seed+ii,int(np.round(onoff_num/num_imgs)),\
                                               centersigma,RFradius,imgs[ii])
        onoffs.append(temp_onoff)
        rms2s.append(temp_rms2)
    onoffs=np.array(onoffs)
    onoffs=onoffs.flatten()
    rms2s=np.array(rms2s)
    rms2s=rms2s.flatten()
    return onoffs,rms2s

#function, get the rms2 image of one image, input: 
#img: image or image region, float, 1.0, could be a little larger than 1.0 for phase scrambled image
#RFradius: the radius of the crop area to be estimated the rms2
#output: rms2_img, float, nonnegative
def rms2_wholeimg(img,RFradius):
    tempdisk=np.float64(disk(RFradius))
    rms2_img=np.zeros((img.shape[0],img.shape[1]))
    for ii in np.arange(RFradius,img.shape[0]-RFradius-1):
        for jj in np.arange(RFradius,img.shape[1]-RFradius-1):
            temp=img[ii-RFradius:ii+RFradius+1,jj-RFradius:jj+RFradius+1]
            temp=temp[np.nonzero(tempdisk)]#circular kernel
            rms2_img[ii,jj]=np.std(temp,ddof=1)/(np.mean(temp)+1e-8)
    if img.shape[0]==437:#whole image frame, not crop
        mask_con=np.zeros((437,437),np.uint8)
        cv2.circle(mask_con,(218,218),radius=218-RFradius,color=255,thickness=-1)
        mask_con=np.float32(mask_con/255.0)
        rms2_img=np.multiply(rms2_img,mask_con)
    return rms2_img

#JSD and bootstrapping
#
#JSD
#calcute the distance between onoff (rms2) distribution in different color channels
#0 means exactly the same, 1 means totally different
#function: jensen-shannon distance between 1d-array x and y
from scipy.stats import entropy
def JSD(x,y,bins):
    data_max=np.max([x,y])
    data_min=np.min([x,y])
    c_x = np.histogram(x,bins,range=[data_min,data_max])[0]
    c_y = np.histogram(y,bins,range=[data_min,data_max])[0]
    c_x = c_x/np.linalg.norm(c_x, ord=1)
    c_y = c_y/np.linalg.norm(c_y, ord=1)
    c_z = 0.5*(c_x+c_y)
    return 0.5 * (entropy(c_x, c_z, base=2) + entropy(c_y, c_z, base=2))
#
#bootstrapping
#apply bootstrapping to estimate standard deviation (error)
#statistics can be offratios, median, mean
#for offratios, be careful with the threshold
#data: for statistics offratios, median, mean: numpy array with shape (sample_size,1)
#num_exp: number of experiments, with replacement
def bootstrap(statistics,data,num_exp=10000,seed=66):
    if   statistics == 'offratios':
        def func(x): return len(x[np.where(x<0)])/len(x[np.where(x>0)]) #threshold is 0, may be different
    elif statistics == 'median':
        def func(x): return np.median(x)
    elif statistics == 'mean':
        def func(x): return np.mean(x)
    elif statistics == 'std':
        def func(x): return np.std(x,ddof=1)
    elif statistics == 'skew':
        def func(x): return skew(x,bias=False)
    #
    sta_boot=np.zeros((num_exp))
    num_data=len(data)
    for ii in range(num_exp):
        np.random.seed(seed+ii)
        tempind=np.random.choice(num_data,num_data,replace=True)
        sta_boot[ii]=func(data[tempind])
    return np.percentile(sta_boot,2.5),np.percentile(sta_boot,97.5)

#apply bootstrapping to estimate standard deviation (error), only for JSD
#data: for statistics JSD: (a,b), a and b are numpy arrays with shape (sample_size,1)
def bootstrap_JSD(data,num_exp=10000,seed=66,bins=64): #bin number 64, may be different
    def func(x0,x1): return JSD(x0,x1,bins) 
    #
    sta_boot=np.zeros((num_exp))
    num_data=len(data[0])
    for ii in range(num_exp):
        np.random.seed(seed+ii)
        tempind=np.random.choice(num_data,num_data,replace=True)
        sta_boot[ii]=func(data[0][tempind],data[1][tempind])
    return np.percentile(sta_boot,2.5),np.percentile(sta_boot,97.5)


#permutation test using monte-carlo method
def perm_test(xs, ys, nmc, randomseed):
    n, k = len(xs), 0
    diff = np.abs(np.median(xs) - np.median(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.seed(randomseed+j)
        np.random.shuffle(zs)
        k += diff < np.abs(np.median(zs[:n]) - np.median(zs[n:]))
    return k / nmc
def perm_test_group(list_of_array,nmc=10000,randomseed=66):
    perm_res=[]
    for ii in np.arange(len(list_of_array)):
        for jj in np.arange(ii+1,len(list_of_array)):
            temp=perm_test(list_of_array[ii], list_of_array[jj], nmc, (ii*jj+jj+randomseed)*nmc)
            perm_res.append(temp)
    return perm_res

