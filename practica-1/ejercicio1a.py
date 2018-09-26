#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

## Ejercicio 1a
def sum_images(img1, img2):
	return img1 + img2

def subtract_images(img1, img2):
	return img1 - img2

def multiply_images(img1, img2):
	return img1 * img2

def main(argv):
	if (len(argv) < 3 or int(argv[0]) > 2 or int(argv[0]) < 0):
		print("Error de parámetros.")
		print("Uso: ejercicio1a.py funcion img1 img2")
		print("Donde, funcion puede ser 0 para suma, 1 para resta ó 2 para multiplicacion")
		print("")
		return

	img1 = cv2.imread(argv[1])
	if (img1 is None):
		print("Error al cargar la imagen 1")
		return

	img2 = cv2.imread(argv[2])
	if (img2 is None):
		print("Error al cargar la imagen 2")
		return

	result = img1
	if (int(argv[0]) == 0):
		result = sum_images(img1, img2)
	elif (int(argv[0]) == 1):
		result = subtract_images(img1, img2)
	else:
		result = multiply_images(img1, img2)

	cv2.namedWindow("window")
	cv2.imshow("window", result)
	cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv[1:])
