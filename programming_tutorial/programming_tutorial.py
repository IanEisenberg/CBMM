#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:10:54 2017

@author: ian
"""
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

input_dim = 50
output_dim = 60
input_vec = np.random.rand(input_dim)*2-1
weight_mat = np.random.rand(output_dim, input_dim)
output_vec = weight_mat.dot(input_vec)

# part 2
def GenerateVoltage(p,T,Vreset,Vthresh,V0):
    V = [V0]
    for i in range(T-1):
        if V[-1]>Vthresh:
            V.append(Vreset)
        else:
            if np.random.rand()<p:
                vd=1
            else:
                vd=-1
            V.append(V[-1]+vd)
    return V

V=GenerateVoltage(.7,1000,-70,-45,-65)
plt.plot(V)
        
# part 3
from scipy.stats import expon

N = 3000
p = 20/1000
spiketrain = (np.random.rand(1,N) < p).flatten()
kernel = expon.pdf(range(-50,50), 5, 10)
output = np.convolve(spiketrain,kernel)
plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(kernel)
plt.subplot(3,1,2)
plt.plot(spiketrain)
plt.subplot(3,1,3)
plt.plot(output)

# part 4
from PIL import Image
from scipy.signal import convolve2d
from skimage.transform import rotate

img = Image.open('octopus.png').convert('L')
k = np.matrix('0 0 0; 0 1.125 0; 0 0 0')-.125
k = k.dot(np.ones([3,3]))
conv_img = convolve2d(img,k)

plt.figure(figsize = (12,14)) 
plt.subplot(3,1,1)
plt.imshow(img)
plt.subplot(3,1,2)
plt.imshow(k)
plt.subplot(3,1,3)
plt.imshow(abs(conv_img))


def create_gabor(): 
    vals = np.linspace(-np.pi,np.pi,50)	
    xgrid, ygrid = np.meshgrid(vals,vals)		
    the_gaussian = np.exp(-(xgrid/2)**2-(ygrid/2)**2)
    # Simple sine wave grating : orientation = 0, phase = 0, amplitude = 1, frequency = 10/(2*pi) 
    the_sine = np.sin(xgrid * 2)	
    # Elementwise multiplication of Gaussian and sine wave grating   
    gabor = the_gaussian * the_sine
    return gabor

rotation = 30
plt.figure(figsize = (12,14)) 
for i, r in enumerate([0,45,90]):
    k = rotate(create_gabor(), r)
    conv_img = convolve2d(img,k)
    plt.subplot(3,2,i*2+1)
    plt.imshow(k)
    plt.subplot(3,2,i*2+2)
    plt.imshow(abs(conv_img))
