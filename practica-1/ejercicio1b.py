#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

## Ejercicio 1b

def scalar_multiply_image(img, scalar):
    return (scalar * img).astype(np.uint8)

def main(argv):
	if (len(argv) < 2):
		print("Error de parámetros.")
		print("Uso: ejercicio1b.py img1 escalar")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return

	s = argv[1]
	try:
		s = float(argv[1])
	except ValueError:
		print("El escalar ingresado no es un número")
		return


	result = scalar_multiply_image(img1, s)

	cv2.namedWindow("window")
	cv2.imshow("window", result)
	cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv[1:])
