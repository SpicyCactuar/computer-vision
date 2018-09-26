#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

## Ejercicio 3

def threshold_image(img, threshold):
    result = img.copy()
    result[result <= threshold] = 0
    result[result > threshold] = 255
    return result

def main(argv):
	if (len(argv) < 2):
		print("Error de parámetros.")
		print("Uso: ejercicio3.py img1 umbral")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return

	t = argv[1]
	try:
		t = int(argv[1])
	except ValueError:
		print("El valor ingresado no es un número")
		return

	if (t < 0 or t > 255):
		print("El valor del umbral debe estar entre 0 y 255")
		return

	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	result = threshold_image(img1, t)

	cv2.namedWindow("window")
	cv2.imshow("window", result)
	cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv[1:])
