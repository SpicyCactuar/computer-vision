#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math
import random

def apply_median_filter(img):
    height, width, _ = img.shape
    result = img.copy()
    
    for i in range(2, height - 2):
        for j in range(2, width - 2):
            result[i, j] = np.median(img[i-2:i+3, j-2:j+3].flatten())
    
    return result

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

def main(argv):
    if (len(argv) < 3):
        print("Error de parámetros.")
        print("Uso: ejercicio6.py img proba-sal proba-pimienta")
        print("img debe ser una imagen en escala de grises")
        print("proba-sal y proba-pimienta en [0.0, 1.0]")
        print("")
        return

    img = cv2.imread(argv[0])

    try:
        saltPercent = float(argv[1])
        pepperPercent = float(argv[2])

        if not (0.0 <= saltPercent and saltPercent <= 1.0 and 0.0 <= pepperPercent and pepperPercent <= 1.0): raise ValueError
    except ValueError:
        print("proba-sal y proba-pimienta en [0.0, 1.0]")
        return

    noisyImg = apply_salt_and_pepper_noise(img, saltPercent, pepperPercent)
    cv2.namedWindow("window")
    cv2.imshow("window", noisyImg)
    cv2.waitKey(0)
    cv2.imshow("window", apply_median_filter(noisyImg))
    cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])