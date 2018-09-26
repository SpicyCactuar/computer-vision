#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

## Ejercicio 8
def equalize_image(img):
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf = cdf / float(cdf.max())

    minimum = cdf.min()
    result = ((cdf - minimum) * 255) / (1 - minimum) + 0.5
    result = result.astype(np.uint8)
    return result[img]

def double_equalize_image(img):
    return equalize_image(equalize_image(img))

## Explicación: La imagen no cambia, en relación a la primera equalización, ya que se intenta distribuir uniformemente algo que ya estaba distribuido uniformemente.

def main(argv):
	if (len(argv) < 1):
		print("Error de parámetros.")
		print("Uso: ejercicio8.py img1")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return

	result = double_equalize_image(img1)

	cv2.namedWindow("window")
	cv2.imshow("window", result)
	cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])
