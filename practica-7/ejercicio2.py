#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2

def main(argv):
    img = cv2.imread("img_target.png")
    if img is None:
        print("No se pudo cargar la imagen img_target.png")
        return
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)
    img = cv2.drawKeypoints(img, kp, None)

    title = "SIFT"
    cv2.namedWindow(title)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
    
if __name__ == '__main__':
    main(sys.argv[1:])