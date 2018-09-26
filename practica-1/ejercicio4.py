#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

## Ejercicio 4

def bit_level_splice(img):
    bit1 = img & 1
    bit2 = img & 2
    bit3 = img & 4
    bit4 = img & 8
    bit5 = img & 16
    bit6 = img & 32
    bit7 = img & 64
    bit8 = img & 128
    return bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8

def main(argv):
	if (len(argv) < 1):
		print("Error de parámetros.")
		print("Uso: ejercicio4.py img1")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return

	results = bit_level_splice(img1)

	for i, img in enumerate(results, 1):
		windowName = "bit" + str(i)
		cv2.namedWindow(windowName)
		cv2.imshow(windowName, img)
		cv2.waitKey(0)
		cv2.destroyWindow(windowName)



if __name__ == '__main__':
    main(sys.argv[1:])
