#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

## Ejercicio 6

def contrast_image(img):
    contraster = lambda pixel: pixel * 0.5 if pixel <= 80 else (pixel if pixel <= 180 else min(pixel * 1.2, 255))
    return np.vectorize(contraster)(img).astype(np.uint8)

def main(argv):
	if (len(argv) < 1):
		print("Error de parámetros.")
		print("Uso: ejercicio6.py img1")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return

	result = contrast_image(img1)

	cv2.namedWindow("window")
	cv2.imshow("window", result)
	cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])
