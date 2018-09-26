#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

## Ejercicio 1c

def scalar_multiply_image(img, scalar):
    return (scalar * img).astype(np.uint8)

def dynamic_range_compression(img):
    L = 256
    R = img.max()
    constant = (L - 1) / math.log10(R + 1)
    transformed_img = np.log10(img + 1)
    return scalar_multiply_image(transformed_img, constant)

def main(argv):
	if (len(argv) < 1):
		print("Error de parámetros.")
		print("Uso: ejercicio1c.py img1")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return


	result = dynamic_range_compression(img1)

	cv2.namedWindow("window")
	cv2.imshow("window", result)
	cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv[1:])
