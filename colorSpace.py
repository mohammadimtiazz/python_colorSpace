# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:28:59 2017

@author: Mohammad Imtiaz
"""


import numpy as np



def RGBtoCMYK(img):
    #declear the following modules before running this function
    #from __future__ import division
    #import numpy as np
    row = len(img)
    col = len(img[0])
    C = np.empty((row, col), dtype = np.uint8)
    M = np.empty((row, col), dtype = np.uint8)
    Y = np.empty((row, col), dtype = np.uint8)
    K = np.empty((row, col), dtype = np.uint8)
    for i in range(0, row):
        for j in range(0, col):
            B = float(int(img[i,j,0])/255.0)
            G = float(int(img[i,j,1])/255.0)
            R = float(int(img[i,j,2])/255.0)
            k = np.min([1 - B, 1 - G,1 - R])
            C[i,j] = ((1 - R - k) / (1 - k)) * 255.0 
            M[i,j] = ((1 - G - k) / (1 - k)) * 255.0
            Y[i,j] = ((1 - B - k) / (1 - k)) * 255.0
            K[i,j] = k * 255.0
            
    return C, M, Y, K 