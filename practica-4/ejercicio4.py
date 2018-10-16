#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math
import random

def supress_non_maximums(corners):
    height, width = corners.shape
    window = 9
    offset = int(math.floor(window / 2))

    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            neighborhood = corners[i - offset : i + offset + 1, j - offset : j + offset + 1]

            if corners[i, j] < neighborhood.max():
                corners[i, j] = 0

    return

def calculate_R(M, method):
    eigenvalues = np.linalg.eigvals(M)
    e1 = np.real(eigenvalues[0])
    e2 = np.real(eigenvalues[1])

    if method == "harris":
        R = e1 * e2 + 0.05 * ((e1 + e2) ** 2)
    elif method == "szeliski":
        R = e1 * e2 / (e1 + e2)
    elif method == "shi-tomasi":
        R = e1
    elif method == "triggs":
        R = e1 - 0.05 * e2

    return R

def detect_corners(img, method, threshold):
    gradientX, gradientY = np.gradient(img.astype(np.float64))
    
    Ixx = cv2.GaussianBlur(gradientX ** 2, (5, 5), 0)
    Iyy = cv2.GaussianBlur(gradientY ** 2, (5, 5), 0)
    Ixy = cv2.GaussianBlur(gradientX * gradientY, (5, 5), 0)

    height, width = img.shape
    window = 3
    offset = int(math.floor(window / 2))

    corners = img.copy().astype(np.float64)

    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            windowIxx = Ixx[i - offset : i + offset + 1, j - offset : j + offset + 1]
            windowIyy = Iyy[i - offset : i + offset + 1, j - offset : j + offset + 1]
            windowIxy = Ixy[i - offset : i + offset + 1, j - offset : j + offset + 1]

            M = np.block([
                [windowIxx.sum(), windowIxy.sum()],
                [windowIxy.sum(), windowIyy.sum()]
            ])

            corners[i, j] = calculate_R(M, method)

    corners[corners < threshold] = 0
    supress_non_maximums(corners)
    corners[corners > 0] = 255

    return corners.astype(np.uint8)

def main(argv):
    if (len(argv) < 3):
        print("Error de parámetros.")
        print("Uso: ejercicio4.py img método umbral")
        print("método = harris, szeliski, shi-tomasi, triggs")
        print("umbral número en [0, +inf)")
        return

    img = cv2.imread(argv[0])
    if img is None:
        print("No se pudo cargar la imagen " + argv[0])
        return

    method = argv[1]
    if not method in ["harris", "szeliski", "shi-tomasi", "triggs"]:
        print("método = harris, szeliski, shi-tomasi, triggs")
        return

    try:
        threshold = int(argv[2])
        if not (0 <= threshold): raise ValueError
    except ValueError:
        print("umbral número en [0, +inf)")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    windowTitle = "Detect Corners with " + method
    cv2.namedWindow(windowTitle)
    cv2.imshow(windowTitle, detect_corners(img, method, threshold))
    cv2.waitKey(0)
    cv2.destroyWindow(windowTitle)
    
if __name__ == '__main__':
    main(sys.argv[1:])