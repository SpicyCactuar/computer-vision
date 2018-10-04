#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

def apply_convolution(img, width, height):
    filteredImg = np.zeros(img.shape, np.uint8)
    boxFilterKernel = np.ones((height, width), np.float64) / (width * height)
    borderHeight = int(math.floor(height / 2.0))
    borderWidth = int(math.floor(width / 2.0))

    for i in range(borderHeight, img.shape[0] - borderHeight):
        for j in range(borderWidth, img.shape[1] - borderWidth):
            filteredImg[i, j] = np.sum(
                img[i - borderHeight : i + borderHeight + 1,
                    j - borderWidth : j + borderWidth + 1] * boxFilterKernel
            )

    return filteredImg

def main(argv):
    if (len(argv) < 3):
        print("Error de parámetros.")
        print("Uso: ejercicio2.py img ancho alto")
        print("img debe ser una imagen en escala de grises")
        print("ancho y alto deben ser números naturales impares")
        print("")
        return

    img = cv2.cvtColor(cv2.imread(argv[0]), cv2.COLOR_BGR2GRAY)
    if (img is None):
        print("Error al cargar la imagen")
        return

    try:
        width = int(argv[1])
        height = int(argv[2])

        if width < 0 or width % 2 == 0 or height < 0 or height % 2 == 0: raise ValueError    
    except ValueError:
        print("ancho y alto deben ser números naturales impares")
        return


    cv2.namedWindow("window")
    cv2.imshow("window", apply_convolution(img, width, height))
    cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])