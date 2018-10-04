#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

gaussianKernel = np.array([
    [1, 4, 7, 4, 1], 
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
]) / 273.0

def smooth_image(img):
    return cv2.filter2D(img, -1, gaussianKernel)

def main(argv):
    imgTest = cv2.imread("../greyscale_images/test.png")
    imgLena = cv2.imread("../greyscale_images/lena.png")

    cv2.namedWindow("window")
    cv2.imshow("window", smooth_image(imgLena))
    cv2.waitKey(0)
    cv2.imshow("window", smooth_image(imgTest))
    cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])