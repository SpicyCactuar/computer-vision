#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plot

## Ejercicio 5

def greyscale_histogram(img):
    counts, bins, bars = plot.hist(img.ravel(), bins=256)
    plot.show()

def main(argv):
	if (len(argv) < 1):
		print("Error de parámetros.")
		print("Uso: ejercicio5.py img1")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return

	greyscale_histogram(img1)

if __name__ == '__main__':
    main(sys.argv[1:])
