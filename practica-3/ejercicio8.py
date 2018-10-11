#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math
import random

sobelKernelXAxis = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

sobelKernelYAxis = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

prewittKernelXAxis = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

prewittKernelYAxis = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

robertsKernelXAxis = np.array([
    [1, 0], 
    [0, -1]
])

robertsKernelYAxis = np.array([
    [0, 1], 
    [-1, 0]
])

def apply_gaussian_noise(img, a, b):
    noisyImage = np.random.normal(a, b, img.shape)
    return (noisyImage + img).astype(np.uint8)

def apply_rayleigh_noise(img, a, b):
    noisyImage = np.random.uniform(0, 1, img.shape)
    noisyImage = a + np.sqrt(-b * np.log(1 - noisyImage))
    return (noisyImage * img).astype(np.uint8)

def apply_salt_and_pepper_noise(img, saltPercent, pepperPercent):
    noisyImg = img.copy()

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            u = random.uniform(0, 1)

            if u < pepperPercent:
                noisyImg[i, j] = 0
            elif u > saltPercent:
                noisyImg[i, j] = 255

    return noisyImg

def detect_borders(img, detector, threshold):
    if detector == "roberts":
        gradientX = cv2.filter2D(img, -1, robertsKernelXAxis)
        gradientY = cv2.filter2D(img, -1, robertsKernelYAxis)
    elif detector == "prewitt":
        gradientX = cv2.filter2D(img, -1, prewittKernelXAxis)
        gradientY = cv2.filter2D(img, -1, prewittKernelYAxis)
    else:
        gradientX = cv2.filter2D(img, -1, sobelKernelXAxis)
        gradientY = cv2.filter2D(img, -1, sobelKernelYAxis)
    
    magnitude = np.sqrt((gradientX.astype(np.float64) ** 2) + (gradientY.astype(np.float64) ** 2))
    result = magnitude.copy()

    result[magnitude <= threshold] = 255
    result[magnitude > threshold] = 0

    return result.astype(np.uint8)

def main(argv):
    if (len(argv) < 3):
        print("Error de parámetros.")
        print("Uso: ejercicio8.py img umbral detector")
        print("umbral un número en [0, 255]")
        print("detector = roberts, prewitt o sobel")
        print("")
        return

    img = cv2.cvtColor(cv2.imread(argv[0]), cv2.COLOR_BGR2GRAY)

    try:
        threshold = int(argv[1])      

        if not (0 <= threshold and threshold <= 255): raise ValueError        
    except ValueError:
        print("umbral un número en [0, 255]")
        return
    try:
        detector = argv[2]
        if detector not in ["sobel", "prewitt", "roberts"]: raise ValueError
    except ValueError:
        print("El detector debe ser sobel, prewitt o roberts")
        return

    cv2.namedWindow("gaussian noise")
    cv2.imshow("gaussian noise", detect_borders(apply_gaussian_noise(img, 10, 10), detector, threshold))
    cv2.waitKey(0)
    cv2.destroyWindow("gaussian noise")

    cv2.namedWindow("rayleigh noise")
    cv2.imshow("rayleigh noise", detect_borders(apply_rayleigh_noise(img, 0.1, 0.5), detector, threshold))
    cv2.waitKey(0)
    cv2.destroyWindow("rayleigh noise")

    cv2.namedWindow("salt and pepper")
    cv2.imshow("salt and pepper", detect_borders(apply_salt_and_pepper_noise(img, 0.95, 0.05), detector, threshold))
    cv2.waitKey(0)
    cv2.destroyWindow("salt and pepper")


if __name__ == '__main__':
    main(sys.argv[1:])