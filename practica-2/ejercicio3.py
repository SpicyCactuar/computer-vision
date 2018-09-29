#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

def split_image_channels(img):
    return cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

def main(argv):
	if (len(argv) < 1):
		print("Error de parámetros.")
		print("Uso: ejercicio3.py img1")
		print("")
		return

	img1 = cv2.imread(argv[0])
	if (img1 is None):
		print("Error al cargar la imagen")
		return

	hue, saturation, value = split_image_channels(img1)

	cv2.namedWindow("hue")
	cv2.imshow("hue", hue)
	cv2.waitKey(0)
	cv2.destroyWindow("hue")

	cv2.namedWindow("saturation")
	cv2.imshow("saturation", saturation)
	cv2.waitKey(0)
	cv2.destroyWindow("saturation")
	
	cv2.namedWindow("value")
	cv2.imshow("value", value)
	cv2.waitKey(0)
	cv2.destroyWindow("value")

if __name__ == '__main__':
    main(sys.argv[1:])

# Sub-ítem a)
# En el canal de 'Value' se observan más detalles que en el resto ya que 'Value' corresponde al nivel de gris que tendría ese píxel en escala de grises.

# Sub-ítem b)
# El canal de 'Hue' es el más afectado por bordes difuminados ya que las partes 'blurreadas', como los bordes difuminados, tienen tonos muy similares. Por ende, se termina viendo más cuadrado/rectangular.