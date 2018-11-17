#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2

def main(argv):
    img = cv2.imread("img_target.png")
    if img is None:
        print("No se pudo cargar la imagen img_target.png")
        return
    
    increment = 45
    angle = increment
    
    rows, cols, _ = img.shape

    while angle < 360:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite("img_target_rotated_" + str(angle) + ".png", dst)
        angle += increment

if __name__ == '__main__':
    main(sys.argv[1:])