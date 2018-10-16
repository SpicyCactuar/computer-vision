#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math
import random

laplacianKernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

def sign(x): return 1 if x > 0 else (0 if x == 0 else -1)

def zero_crossing(img):
    # Do the following for each pixel I(u,v)
    #     1. Look at your four neighbors, left, right, up and down
    #     2. If they all have the same sign as you, then you are not a zero crossing
    #     3. Else, if you have the smallest absolute value compared to your neighbors with opposite sign, then you are a zero crossing
    height, width = img.shape
    zeroCrossings = np.zeros(img.shape)

    for i in range(0, height):
        for j in range(0, width):
            neighbors = []
            if i > 0:
                neighbors.append(img[i-1, j])
            if i < height-1:
                neighbors.append(img[i+1, j])
            if j > 0:
                neighbors.append(img[i, j-1])
            if j < width-1:
                neighbors.append(img[i, j+1])
            
            neighbors = np.array(neighbors)
            mySign = sign(img[i, j])
            neighborsSign = np.sign(neighbors)
            if np.all(neighborsSign == mySign):
                zeroCrossings[i, j] = 0
            else:
                neighbors = np.abs(neighbors[neighborsSign != mySign])
                if np.any(neighbors < math.fabs(img[i, j])):
                    zeroCrossings[i, j] = 0
                else: 
                    zeroCrossings[i, j] = 255
    
    return zeroCrossings.astype(np.uint8)


def laplacianOfGaussian(img):
    result = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float64)
    result = cv2.filter2D(result, -1, laplacianKernel)
    return result

def main(argv):
    if (len(argv) < 1):
        print("Error de parámetros.")
        print("Uso: ejercicio2c.py img")
        print("")
        return

    img = cv2.imread(argv[0])
    if img is None:
        print("No se pudo cargar la imagen " + argv[0])
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.namedWindow("Laplace")
    cv2.imshow("Laplace", laplacianOfGaussian(img))
    cv2.waitKey(0)
    cv2.destroyWindow("Laplace")
    
if __name__ == '__main__':
    main(sys.argv[1:])