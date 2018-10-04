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

def sharpen_image(img):
    smoothImage = cv2.filter2D(img, -1, gaussianKernel)
    return img - smoothImage + img

def main(argv):
    imgTest = cv2.imread("../greyscale_images/test.png")
    imgLena = cv2.imread("../greyscale_images/lena.png")

    cv2.namedWindow("window")
    cv2.imshow("window", sharpen_image(imgLena))
    cv2.waitKey(0)
    cv2.imshow("window", sharpen_image(imgTest))
    cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])