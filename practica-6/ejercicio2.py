#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math

def calculateAi(correspondence):
    p1, p2 = correspondence
    x2, y2, w2 = p2

    return np.block([
        [0, 0, 0, -w2 * p1, y2 * p1],
        [w2 * p1, 0, 0, 0, -x2 * p1]
    ])

def DLT(correspondences):
    A = []
    for correspondence in correspondences:
        A.append(calculateAi(correspondence))
    A = np.concatenate(A)

    _, _, Vt = np.linalg.svd(A)
    V = Vt.transpose()
    return V[:, 8].reshape(3, 3)


def RANSAC(correspondences):

    correspondences = np.array(correspondences)
    # Suponiendo 5% de outliers
    T = 0.95 * len(correspondences)
    N = 3
    t = math.sqrt(5.99 * 15) # asumimos varianza 15

    for j in range(N):
        homographies = []

        consensusNumber = 0
        s = correspondences[np.random.choice(correspondences.shape[0], 4, replace=False), :]
        H = DLT(s)
        firstImagePoints = s[:, 0]
        for i in range(len(firstImagePoints)):
            newPoint = H*firstImagePoints[i]
            if np.linalg.norm(newPoint - s[i, 1]) < t:
                consensusNumber += 1
        homographies.append((H, consensusNumber))

        if consensusNumber > T:
            return H
    
    homographies.sort(key= lambda x: x[1])
    return homographies[0][0]


def main(argv):
    img = cv2.imread("dameros/damero2.jpg")
    if img is None:
        print("No se pudo cargar la imagen dameros/damero2.jpg")
        return

    # TODO: Encontrar correspondencias correctas
    correspondences = [
        np.array([[867, 780, 1], [315, 1225, 1]]),
        np.array([[864, 1078, 1], [472, 1457, 1]]),
        np.array([[1160, 1085, 1], [710, 1281, 1]]),
        np.array([[1162, 783, 1], [550, 1047, 1]])
    ]

    rows, cols, _ = img.shape
    H = RANSAC(correspondences)
    result = cv2.warpPerspective(img, H, (cols, rows))

    cv2.namedWindow("RANSAC")
    cv2.imshow("RANSAC", result)
    cv2.waitKey(0)
    cv2.destroyWindow("RANSAC")
    cv2.imwrite("dameroTransformado_RANSAC.png", result)
    
if __name__ == '__main__':
    main(sys.argv[1:])