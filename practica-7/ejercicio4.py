#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import glob

def main(argv):

    files = glob.glob("img_target_*.png")

    if len(files) == 0:
        print("Correr primero los ejercicio3x.py")
        return

    for filename in files:
        img = cv2.imread(filename)
        if img is None:
            print("No se pudo cargar la imagen " + filename)
            return
        
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(img, None)
        img = cv2.drawKeypoints(img, kp, None)

        title = filename
        cv2.namedWindow(title)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
        
if __name__ == '__main__':
    main(sys.argv[1:])