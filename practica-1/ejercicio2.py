#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

## Ejercicio 2

def negative_image(img):
    return 255 - img

def main(argv):
	if (len(argv) < 1):
		print("Error de parámetros.")
		print("Uso: ejercicio2.py img1")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return


	result = negative_image(img1)

	cv2.namedWindow("window")
	cv2.imshow("window", result)
	cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv[1:])
