#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

def linear_saturation_transformation(img):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsvImg[:,:,1] += 80
    return cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

def logarithmic_saturation_transformation(img):
	hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
	hsvImg[:,:,1] = np.log10(hsvImg[:,:,1])
	return cv2.cvtColor(hsvImg.astype(np.uint8), cv2.COLOR_HSV2BGR)

def main(argv):
	if (len(argv) < 2):
		print("Error de parámetros.")
		print("Uso: ejercicio1b.py img1 eleccion")
		print("eleccion = lineal o no-lineal")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return

	choice = argv[1]
	if choice != "lineal" and choice != "no-lineal":
		print("El parámetro de elección no es válido")
		print("eleccion = lineal o no-lineal")
		return

	if choice == "lineal":
		result = linear_saturation_transformation(img1)
	else:
		result = logarithmic_saturation_transformation(img1)

	cv2.namedWindow("window")
	cv2.imshow("window", result)
	cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv[1:])