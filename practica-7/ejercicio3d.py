#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

def main(argv):
    img = cv2.imread("img_target.png")
    if img is None:
        print("No se pudo cargar la imagen img_target.png")
        return

    increment = 10
    blur = increment

    while blur <= 40:
        dst = cv2.GaussianBlur(img, (5, 5), blur)
        cv2.imwrite("img_target_blurred_" + str(blur) + ".png", dst)
        blur += increment

if __name__ == '__main__':
    main(sys.argv[1:])