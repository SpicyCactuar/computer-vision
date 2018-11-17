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
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    increment = 32
    brightness = -127

    while brightness <= 127:
        dst = img.copy()
        dst[:, :, 2] = np.clip(dst[:, :, 2].astype(np.int16) + brightness, 0, 255).astype(np.uint8)
        cv2.imwrite("img_target_brightened_" + str(brightness) + ".png", cv2.cvtColor(dst, cv2.COLOR_HSV2BGR))
        brightness += increment

if __name__ == '__main__':
    main(sys.argv[1:])