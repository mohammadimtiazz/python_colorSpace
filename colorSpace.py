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


def RGBtoXYZ(img):
    # Observer= 2°, Illuminant= D65
    row = img.shape[0]
    col = img.shape[1]
    X = np.empty((row, col), dtype = np.uint8)
    Y = np.empty((row, col), dtype = np.uint8)
    Z = np.empty((row, col), dtype = np.uint8)
    var_B = (img[:,:,0])/255.0
    var_G = (img[:,:,1])/255.0
    var_R = (img[:,:,2])/255.0
    
    indicesVarB = np.where(var_B > 0.04045)
    indicesVarBnot = np.where(var_B <= 0.04045)
    var_B[indicesVarB[0][:], indicesVarB[1][:]] = (((var_B[indicesVarB[0][:], indicesVarB[1][:]]+0.055)/1.055)**2.4)*100
    var_B[indicesVarBnot[0][:], indicesVarBnot[1][:]] = (var_B[indicesVarBnot[0][:], indicesVarBnot[1][:]]/12.92)*100
    
    indicesVarG = np.where(var_G > 0.04045)
    indicesVarGnot = np.where(var_G <= 0.04045)
    var_G[indicesVarG[0][:], indicesVarG[1][:]] = (((var_G[indicesVarG[0][:], indicesVarG[1][:]]+0.055)/1.055)**2.4)*100
    var_G[indicesVarGnot[0][:], indicesVarGnot[1][:]] = (var_G[indicesVarGnot[0][:], indicesVarGnot[1][:]]/12.92)*100    
    
    indicesVarR = np.where(var_R > 0.04045)
    indicesVarRnot = np.where(var_R <= 0.04045)
    var_R[indicesVarR[0][:], indicesVarR[1][:]] = (((var_R[indicesVarR[0][:], indicesVarR[1][:]]+0.055)/1.055)**2.4)*100
    var_R[indicesVarRnot[0][:], indicesVarRnot[1][:]] = (var_R[indicesVarRnot[0][:], indicesVarRnot[1][:]]/12.92)*100   
    
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
    
    return X,Y,Z



    
def RGBtoLUV(img):
    row = img.shape[0]
    col = img.shape[1]
    CIEL = np.empty((row, col), dtype = np.uint8)
    CIEU = np.empty((row, col), dtype = np.uint8)
    CIEV = np.empty((row, col), dtype = np.uint8)    
    
    XYZimg = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    X = XYZimg[:,:,0]
    Y = XYZimg[:,:,1]
    Z = XYZimg[:,:,2]
    
    var_U = ( 4 * X ) / ( X + ( 15 * Y ) + ( 3 * Z ) )
    var_V = ( 9 * Y ) / ( X + ( 15 * Y ) + ( 3 * Z ) )
    
    var_Y = Y / 100.0
    
    indicesVarY = np.where(var_Y > 0.008856)
    indicesVarYnot = np.where(var_Y <= 0.008856)
    var_Y[indicesVarY[0][:], indicesVarY[1][:]] = (var_Y[indicesVarY[0][:], indicesVarY[1][:]]) ** (1.0/3.0)
    var_Y[indicesVarYnot[0][:], indicesVarYnot[1][:]] = (7.787 * var_Y[indicesVarYnot[0][:], indicesVarYnot[1][:]]) + (16.0/116.0)   
    
    ref_X =  95.047        #Observer= 2°, Illuminant= D65
    ref_Y = 100.000
    ref_Z = 108.883
    
    ref_U = ( 4 * ref_X ) / ( ref_X + ( 15 * ref_Y ) + ( 3 * ref_Z ) )
    ref_V = ( 9 * ref_Y ) / ( ref_X + ( 15 * ref_Y ) + ( 3 * ref_Z ) )
    
    CIEL = ( 116 * var_Y ) - 16
    CIEU = 13 * CIEL * ( var_U - ref_U )
    CIEV = 13 * CIEL * ( var_V - ref_V )
    
    return CIEL, CIEU, CIEV



    
