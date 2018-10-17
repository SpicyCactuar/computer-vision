#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math
import random

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

def detect_borders(img, threshold):
    gradientX = cv2.filter2D(img.astype(np.float64), -1, prewittKernelXAxis)
    gradientY = cv2.filter2D(img.astype(np.float64), -1, prewittKernelYAxis)
    magnitude = np.sqrt((gradientX.astype(np.float64) ** 2) + (gradientY.astype(np.float64) ** 2))
    result = magnitude.copy()

    result[magnitude <= threshold] = 255
    result[magnitude > threshold] = 0

    return result.astype(np.uint8)

def main(argv):
    if (len(argv) < 2):
        print("Error de parámetros.")
        print("Uso: ejercicio7b.py img umbral")
        print("umbral un número en [0, 255]")
        print("")
        return

    img = cv2.cvtColor(cv2.imread(argv[0]), cv2.COLOR_BGR2GRAY)

    try:
        threshold = int(argv[1])

        if not (0 <= threshold and threshold <= 255): raise ValueError
    except ValueError:
        print("umbral un número en [0, 255]")
        return

    cv2.namedWindow("window")
    cv2.imshow("window", detect_borders(img, threshold))
    cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])
