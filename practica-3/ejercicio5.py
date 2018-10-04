#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

def apply_median_filter(img):
    height, width, _ = img.shape
    result = img.copy()
    
    for i in range(2, height - 2):
        for j in range(2, width - 2):
            result[i, j] = np.median(img[i-2:i+3, j-2:j+3].flatten())
    
    return result

def main(argv):
    if (len(argv) < 1):
        print("Error de parámetros.")
        print("Uso: ejercicio5.py img")
        print("")
        return

    img = cv2.imread(argv[0])

    cv2.namedWindow("window")
    cv2.imshow("window", apply_median_filter(img))
    cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])