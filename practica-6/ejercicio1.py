#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

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

def main(argv):
    img = cv2.imread("dameros/damero2.jpg")
    if img is None:
        print("No se pudo cargar la imagen dameros/damero2.jpg")
        return

    correspondences25 = [
        np.array([[867, 780, 1], [315, 1225, 1]]),
        np.array([[864, 1078, 1], [472, 1457, 1]]),
        np.array([[1160, 1085, 1], [710, 1281, 1]]),
        np.array([[1162, 783, 1], [550, 1047, 1]])
    ]
    rows, cols, _ = img.shape
    H = DLT(correspondences25)
    result = cv2.warpPerspective(img, H, (cols, rows))

    cv2.namedWindow("DLT")
    cv2.imshow("DLT", result)
    cv2.waitKey(0)
    cv2.destroyWindow("DLT")
    cv2.imwrite("dameroTransformado_DLT.png", result)
    
if __name__ == '__main__':
    main(sys.argv[1:])