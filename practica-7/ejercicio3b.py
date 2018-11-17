#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2

def main(argv):
    img = cv2.imread("img_target.png")
    if img is None:
        print("No se pudo cargar la imagen img_target.png")
        return
    
    increment = 0.25
    scaleFactor = increment

    while scaleFactor <= 2.0:
        dst = cv2.resize(img, None, fx=scaleFactor, fy=scaleFactor, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("img_target_scaled_" + str(scaleFactor) + ".png", dst)
        scaleFactor += increment

if __name__ == '__main__':
    main(sys.argv[1:])