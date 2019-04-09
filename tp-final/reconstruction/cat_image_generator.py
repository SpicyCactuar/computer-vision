#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import random
import numpy as np
import glob
import csv

perturbation_range = 16

def prepare_image_for_dataset(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (640, 480))
    pX = 287
    pY = 114

    crop = image[pY : pY + 128, pX : pX + 128].copy()

    original_points = np.float32([
        (pX, pY),
        (pX + 128, pY),
        (pX + 128, pY + 128),
        (pX, pY + 128)
    ])
    perturbed_points = np.float32([
        perturb(pX, pY),
        perturb(pX + 128, pY),
        perturb(pX + 128, pY + 128),
        perturb(pX, pY + 128)
    ])

    four_point_homography = perturbed_points - original_points
    four_point_homography = four_point_homography.flatten()
    homography = cv2.getPerspectiveTransform(original_points, perturbed_points)

    perturbed_image = cv2.warpPerspective(image, homography, (640, 480), flags=cv2.WARP_INVERSE_MAP)

    # cv2.namedWindow("window")
    # cv2.imshow("window", perturbed_image)
    # cv2.waitKey(0)

    perturbed_crop = perturbed_image[pY : pY + 128, pX : pX + 128].copy()

    return (crop, perturbed_crop, four_point_homography)

def perturb(x, y):
    return (x + random.randint(-perturbation_range, perturbation_range),
            y + random.randint(-perturbation_range, perturbation_range))

def main():
    crop, perturbed_crop, homography = prepare_image_for_dataset("cat_1.png")

    cv2.imwrite("cat_original.png", crop)
    cv2.imwrite("cat_perturbed.png", perturbed_crop)

    with open("cat.csv", mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(homography)

if __name__ == "__main__":
    main()
