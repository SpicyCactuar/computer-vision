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

    for _ in range(N):
        homographies = []

        consensusNumber = 0
        s = correspondences[np.random.choice(len(correspondences), 4, replace=False), :]
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
    damero2 = cv2.imread("dameros/damero2.jpg")
    damero5 = cv2.imread("dameros/damero5.jpg")
    if damero2 is None:
        print("No se pudo cargar la imagen dameros/damero2.jpg")
        return
    if damero5 is None:
        print("No se pudo cargar la imagen dameros/damero5.jpg")
        return

    sift = cv2.xfeatures2d.SIFT_create()
    damero2Kp, damero2Des = sift.detectAndCompute(damero2, None)
    damero5Kp, damero5Des = sift.detectAndCompute(damero5, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(damero2Des, damero5Des, k=2)

    rows, cols, _ = damero2.shape
    srcPoints = np.array([damero2Kp[mat[0].queryIdx].pt + (1,) for mat in matches])
    dstPoints = np.array([damero5Kp[mat[0].trainIdx].pt + (1,) for mat in matches])
    correspondences = list(zip(srcPoints, dstPoints))
    H = RANSAC(correspondences)
    result = cv2.warpPerspective(damero2, H, (cols, rows))

    cv2.namedWindow("RANSAC")
    cv2.imshow("RANSAC", result)
    cv2.waitKey(0)
    cv2.destroyWindow("RANSAC")
    cv2.imwrite("dameroTransformado_RANSAC.png", result)
    
if __name__ == '__main__':
    main(sys.argv[1:])