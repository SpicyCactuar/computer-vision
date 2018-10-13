#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math
import random

sobelKernelXAxis = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

sobelKernelYAxis = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

prewittKernelXAxis = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

prewittKernelYAxis = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

robertsKernelXAxis = np.array([
    [1, 0], 
    [0, -1]
])

robertsKernelYAxis = np.array([
    [0, 1], 
    [-1, 0]
])

def apply_gaussian_noise(img, a, b):
    noisyImage = np.random.normal(a, b, img.shape)
    return (noisyImage + img).astype(np.uint8)

def apply_rayleigh_noise(img, a, b):
    noisyImage = np.random.uniform(0, 1, img.shape)
    noisyImage = a + np.sqrt(-b * np.log(1 - noisyImage))
    return (noisyImage * img).astype(np.uint8)

def apply_salt_and_pepper_noise(img, saltPercent, pepperPercent):
    noisyImg = img.copy()

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            u = random.uniform(0, 1)

            if u < pepperPercent:
                noisyImg[i, j] = 0
            elif u > saltPercent:
                noisyImg[i, j] = 255

    return noisyImg

def calculate_gradients(img, detector):
    blurredImg = cv2.GaussianBlur(img, (5,5), 0).astype(np.float64)

    if detector == "roberts":
        gradientX = cv2.filter2D(blurredImg, -1, robertsKernelXAxis)
        gradientY = cv2.filter2D(blurredImg, -1, robertsKernelYAxis)
    elif detector == "prewitt":
        gradientX = cv2.filter2D(blurredImg, -1, prewittKernelXAxis)
        gradientY = cv2.filter2D(blurredImg, -1, prewittKernelYAxis)
    else:
        gradientX = cv2.filter2D(blurredImg, -1, sobelKernelXAxis)
        gradientY = cv2.filter2D(blurredImg, -1, sobelKernelYAxis)

    gradientX = gradientX.astype(np.float64)
    gradientY = gradientY.astype(np.float64)

    magnitude = np.sqrt((gradientX ** 2) + (gradientY ** 2))
    angles = np.rad2deg(np.arctan2(gradientY, gradientX))

    return (magnitude, angles)

def closest_direction(angle):
    if angle < 22.5:
        return 0.0
    elif angle < (45 + 22.5):
        return 45.0
    elif angle < (90 + 22.5):
        return 90.0
    elif angle < (135 + 22.5):
        return 135.0
    else:
        return 0

def border_value(magnitude, i, j, direction):
    height, width = magnitude.shape
    neighbors = []
    if direction == 0.0:
        if j < width - 1:
            neighbors.append(magnitude[i, j + 1])
        if j > 0:
            neighbors.append(magnitude[i, j - 1])
    elif direction == 45.0:
        if j < width - 1 and i > 0:
            neighbors.append(magnitude[i - 1, j + 1])
        if j > 0 and i < height - 1:
            neighbors.append(magnitude[i + 1, j - 1])
    elif direction == 90.0:
        if i < height - 1:
            neighbors.append(magnitude[i + 1, j])
        if i > 0:
            neighbors.append(magnitude[i - 1, j])
    elif direction == 135.0:
        if j < width - 1 and i < height - 1:
            neighbors.append(magnitude[i + 1, j + 1])
        if j > 0 and i > 0:
            neighbors.append(magnitude[i - 1, j - 1])

    magnitudeValue = magnitude[i, j]

    for neighbor in neighbors:
        if magnitudeValue < neighbor:
            return 0

    return magnitudeValue
    

def supress_non_maximums(magnitude, angles):
    normalizedAngles = angles.copy()
    normalizedAngles[angles < 0.0] += 180.0

    result = magnitude.copy()
    height, width = angles.shape

    for i in range(0, height):
        for j in range(0, width):
            direction = closest_direction(normalizedAngles[i, j])
            borderValue = border_value(magnitude, i, j, direction)
            result[i, j] = borderValue

    return result

def traversing_directions(position, direction, shape):
    height, width = shape
    directions = []
    i, j = position
    
    if direction == 0.0:
        if i < height - 1:
            directions.append((1, 0))
        if i > 0:
            directions.append((-1, 0))
    elif direction == 45.0:
        if j < width - 1 and i < height - 1:
            directions.append((1, 1))
        if j > 0 and i > 0:
            directions.append((-1, -1))
    elif direction == 90.0:
        if j < width - 1:
            directions.append((0, 1))
        if j > 0:
            directions.append((0, -1))
    elif direction == 135.0:
        if j < width - 1 and i > 0:
            directions.append((-1, 1))
        if j > 0 and i < height - 1:
            directions.append((1, -1))

    return directions

def traverse_border(position, magnitude, angles, bordersMask, uMin):
    pendingPositions = [position]

    while len(pendingPositions) > 0:
        currentPosition = pendingPositions.pop()
        traversingDirections = traversing_directions(currentPosition, closest_direction(angles[currentPosition]), magnitude.shape)

        for direction in traversingDirections:
            nextPosition = (currentPosition[0] + direction[0], currentPosition[1] + direction[1])
            if bordersMask[nextPosition]: continue
            
            bordersMask[nextPosition] = magnitude[nextPosition] > uMin

            if bordersMask[nextPosition]:
                pendingPositions.append(nextPosition)

    return

def hysteresis_thresholding(magnitude, angles, uMin, uMax):
    bordersMask = magnitude > uMax
    normalizedAngles = angles.copy()
    normalizedAngles[angles < 0.0] += 180.0
    height, width = magnitude.shape

    for i in range(0, height):
        for j in range(0, width):
            if not bordersMask[i, j]: continue
            traverse_border((i, j), magnitude, normalizedAngles, bordersMask, uMin)

    result = magnitude.copy()
    result[np.logical_not(bordersMask)] = 0.0
    result[bordersMask] = 255.0
    return result

def detect_borders(img, detector, uMin, uMax, hysteresis=True):
    magnitude, angles = calculate_gradients(img, detector)
    supressedMagnitude = supress_non_maximums(magnitude, angles)
    if not hysteresis:
        return supressedMagnitude.astype(np.uint8)

    borders = hysteresis_thresholding(supressedMagnitude, angles, uMin, uMax)
    return borders.astype(np.uint8)

def main(argv):
    if (len(argv) < 3):
        print("Error de parámetros.")
        print("Uso: ejercicio3.py u-min u-max detector")
        print("u-min y u-max números en [0, 255]")
        print("detector = roberts, prewitt o sobel")
        print("")
        return

    imgLena = cv2.cvtColor(cv2.imread("lena.png"), cv2.COLOR_BGR2GRAY)
    imgTest = cv2.cvtColor(cv2.imread("test.png"), cv2.COLOR_BGR2GRAY)

    try:
        uMin = float(argv[0])
        uMax = float(argv[1])

        if not (0.0 <= uMin and uMin <= 255.0): raise ValueError
        if not (0.0 <= uMax and uMax <= 255.0): raise ValueError
    except ValueError:
        print("u-min y u-max números en [0, 255]")
        return

    try:
        detector = argv[2]
        if detector not in ["sobel", "prewitt", "roberts"]: raise ValueError
    except ValueError:
        print("El detector debe ser sobel, prewitt o roberts")
        return

    cv2.namedWindow("Original image")
    cv2.imshow("Original image", imgTest)
    cv2.waitKey(0)
    cv2.destroyWindow("Original image")

    # Sin histéresis
    cv2.namedWindow("Non-maximum supression")
    cv2.imshow("Non-maximum supression", detect_borders(imgTest, detector, uMin, uMax, False))
    cv2.waitKey(0)
    cv2.destroyWindow("Non-maximum supression")

    cv2.namedWindow("gaussian noise")
    cv2.imshow("gaussian noise", detect_borders(apply_gaussian_noise(imgTest, 10, 10), detector, uMin, uMax, False))
    cv2.waitKey(0)
    cv2.destroyWindow("gaussian noise")

    cv2.namedWindow("rayleigh noise")
    cv2.imshow("rayleigh noise", detect_borders(apply_rayleigh_noise(imgTest, 0.1, 0.5), detector, uMin, uMax, False))
    cv2.waitKey(0)
    cv2.destroyWindow("rayleigh noise")

    cv2.namedWindow("salt and pepper")
    cv2.imshow("salt and pepper", detect_borders(apply_salt_and_pepper_noise(imgTest, 0.95, 0.05), detector, uMin, uMax, False))
    cv2.waitKey(0)
    cv2.destroyWindow("salt and pepper")

    # Con histéresis
    cv2.namedWindow("With hysteresis")
    cv2.imshow("With hysteresis", detect_borders(imgTest, detector, uMin, uMax, True))
    cv2.waitKey(0)
    cv2.destroyWindow("With hysteresis")

    cv2.namedWindow("gaussian noise")
    cv2.imshow("gaussian noise", detect_borders(apply_gaussian_noise(imgTest, 10, 10), detector, uMin, uMax))
    cv2.waitKey(0)
    cv2.destroyWindow("gaussian noise")

    cv2.namedWindow("rayleigh noise")
    cv2.imshow("rayleigh noise", detect_borders(apply_rayleigh_noise(imgTest, 0.1, 0.5), detector, uMin, uMax))
    cv2.waitKey(0)
    cv2.destroyWindow("rayleigh noise")

    cv2.namedWindow("salt and pepper")
    cv2.imshow("salt and pepper", detect_borders(apply_salt_and_pepper_noise(imgTest, 0.95, 0.05), detector, uMin, uMax))
    cv2.waitKey(0)
    cv2.destroyWindow("salt and pepper")

if __name__ == '__main__':
    main(sys.argv[1:])