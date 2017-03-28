# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:28:59 2017

@author: Mohammad Imtiaz
"""


import numpy as np
import cv2

def searchValue(inputArray, value):
    tupleArray = np.where(inputArray == value)
    outputArray = np.transpose(tupleArray)
    return outputArray


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


def RGBtoCMYKfast(img):
    #declear the following modules before running this function
    #from __future__ import division
    #import numpy as np
    row = img.shape[0]
    col = img.shape[1]
    C = np.empty((row, col), dtype = np.uint8)
    M = np.empty((row, col), dtype = np.uint8)
    Y = np.empty((row, col), dtype = np.uint8)
    K = np.empty((row, col), dtype = np.uint8)
    B = (img[:,:,0])/255.0
    G = (img[:,:,1])/255.0
    R = (img[:,:,2])/255.0
    kb = 1.0 - B
    kg = 1.0 - G
    kr = 1.0 - R
    kbg = np.minimum(kb,kg)
    k = np.minimum(kbg,kr)   
    checkForOne = searchValue(k, 1)
    C = np.uint8(((1.0 - R - k) / (1.0 - k)) * 255 )
    M = np.uint8(((1.0 - G - k) / (1.0 - k)) * 255)
    Y = np.uint8(((1.0 - B - k) / (1.0 - k)) * 255)
    K = np.uint8(k * 255  )
    if checkForOne.size == 0:
        C[checkForOne[:,0], checkForOne[:,1]] = 0
        M[checkForOne[:,0], checkForOne[:,1]] = 0
        Y[checkForOne[:,0], checkForOne[:,1]] = 0
        K[checkForOne[:,0], checkForOne[:,1]] = 1
    else:
        pass       
            
    return C, M, Y, K 
