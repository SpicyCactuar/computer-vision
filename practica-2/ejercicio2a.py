#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

def sum_hue(img, factor):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsvImg[:,:,0] += factor
    return cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

def main(argv):
	if (len(argv) < 2):
		print("Error de parámetros.")
		print("Uso: ejercicio2a.py img1 c")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return

	try:
		c = int(argv[1])
	except ValueError:
		print("El valor ingresado no es un número entero")
		return

	result = sum_hue(img1, c)

	cv2.namedWindow("window")
	cv2.imshow("window", result)
	cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])