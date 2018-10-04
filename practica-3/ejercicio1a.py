#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

lowPassSmallKernel = np.array([
    [1, 2, 1], 
    [2, 4, 2], 
    [1, 2, 1]
]) / 16.0

highPassSmallKernel = np.array([
    [1, -2, 1],
    [-2, 4, -2],
    [1, -2, 1]
]) / 4.0

lowPassBigKernel = np.array([
    [1, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
]) / 25.0

highPassBigKernel = np.array([
    [-2, -1, 0, 1, 2], 
    [-4, -2, 0, 2, 4],
    [-8, -4, 0, 4, 8],
    [-4, -2, 0, 2, 2],
    [-2, -1, 0, 1, 4]
]) / 60.0


def apply_convolution(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def main(argv):
    if (len(argv) < 3):
        print("Error de parámetros.")
        print("Uso: ejercicio1a.py img1 tipo tamaño")
        print("tipo = pasa-bajo o pasa-alto")
        print("tamaño = 3 o 5")
        print("")
        return

    img1 = cv2.imread(argv[0])
    if (img1 is None):
        print("Error al cargar la imagen")
        return

    filterType = argv[1]
    if filterType != "pasa-bajo" and filterType != "pasa-alto":
        print("El parámetro de tipo no es válido")
        print("tipo = pasa-bajo o pasa-alto")
        return
    
    kernelSize = argv[2]
    if kernelSize != "3" and kernelSize != "5":
        print("El parámetro de tamaño no es válido")
        print("tamaño = 3 o 5")
        return
        
    if filterType == "pasa-bajo":
        kernel = lowPassSmallKernel if kernelSize == "3" else lowPassBigKernel
        result = apply_convolution(img1, kernel)
    else:
        kernel = highPassSmallKernel if kernelSize == "3" else highPassBigKernel
        result = apply_convolution(img1, kernel)

    cv2.namedWindow("window")
    cv2.imshow("window", result)
    cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])