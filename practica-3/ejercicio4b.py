#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math
import random

gaussianKernel = np.array([
    [1, 4, 7, 4, 1], 
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
]) / 273.0

def apply_rayleigh_noise(img, a, b):
    noisyImage = np.random.uniform(0, 1, img.shape)
    noisyImage = a + np.sqrt(-b * np.log(1 - noisyImage))
    return noisyImage.astype(np.uint8) + img

def smooth_image_with_noise(img, a, b):
    return cv2.filter2D(apply_rayleigh_noise(img, a, b), -1, gaussianKernel)

def sharpen_image_with_noise(img, a, b):
    noisyImage = apply_rayleigh_noise(img, a, b)
    smoothImage = cv2.filter2D(noisyImage, -1, gaussianKernel)
    return noisyImage - smoothImage + noisyImage

def main(argv):
    if (len(argv) < 2):
        print("Error de parámetros.")
        print("Uso: ejercicio4b.py a b")
        print("a, b = número natural")
        print("")
        return

    try:
        a = int(argv[0])
        b = int(argv[1])

        if a < 0 or b < 0: raise ValueError
    except ValueError:
        print("Los valores a y b tienen que ser números naturales")
        return

    imgTest = cv2.cvtColor(cv2.imread("../greyscale_images/test.png"), cv2.COLOR_BGR2GRAY)
    imgLena = cv2.cvtColor(cv2.imread("../greyscale_images/lena.png"), cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("window")
    cv2.imshow("window", smooth_image_with_noise(imgLena, a, b))
    cv2.waitKey(0)
    cv2.imshow("window", smooth_image_with_noise(imgTest, a, b))
    cv2.waitKey(0)
    cv2.imshow("window", sharpen_image_with_noise(imgLena, a, b))
    cv2.waitKey(0)
    cv2.imshow("window", sharpen_image_with_noise(imgTest, a, b))
    cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])