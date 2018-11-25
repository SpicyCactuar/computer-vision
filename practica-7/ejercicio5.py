#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import glob
import numpy as np

def main(argv):

    originalImg = cv2.imread("img_target.png")
    if originalImg is None:
        print("No se pudo cargar la imagen img_target.png")
        return

    sift = cv2.xfeatures2d.SIFT_create()
    originalKp, originalDes = sift.detectAndCompute(originalImg, None)

    files = glob.glob("img_target_*.png")
    if len(files) == 0:
        print("Correr primero los ejercicio3x.py")
        return

    for filename in files:
        img = cv2.imread(filename)
        if img is None:
            print("No se pudo cargar la imagen " + filename)
            return
        
        kp, des = sift.detectAndCompute(img, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(originalDes, des, k=2)
        goodMatches = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatches.append([m])

        srcPoints = np.array([originalKp[mat[0].queryIdx].pt for mat in goodMatches])
        dstPoints = np.array([kp[mat[0].trainIdx].pt for mat in goodMatches])
        _, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        mask = mask.ravel().tolist()
        goodMatches = [match[0] for match in goodMatches]
        goodMatches = np.ma.array(goodMatches, mask=mask).compressed()
        goodMatches = [[match] for match in goodMatches]

        matchImg = cv2.drawMatchesKnn(originalImg, originalKp, img, kp, goodMatches, outImg=None, flags=2)

        print(filename + " Nm/Nk: " + str(len(goodMatches) / float(len(originalKp)))) 

        title = filename
        cv2.namedWindow(title)
        cv2.imshow(title, matchImg)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
        
if __name__ == '__main__':
    main(sys.argv[1:])