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
    pX = random.randint(perturbation_range, 640 - 128 - perturbation_range)
    pY = random.randint(perturbation_range, 480 - 128 - perturbation_range)

    crop = image[pX : pX + 128, pY : pY + 128].copy()

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
    homography = cv2.getPerspectiveTransform(original_points, perturbed_points)
    # Ver si estamos consiguiendo la inversa o no

    perturbed_image = cv2.warpPerspective(image, homography, (640, 480), flags=cv2.WARP_INVERSE_MAP)
    perturbed_crop = perturbed_image[pX : pX + 128, pY : pY + 128].copy()

    print("Original")
    print(original_points)
    print("Perturbed")
    print(perturbed_points)

    cv2.namedWindow("window")
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.imshow("window", perturbed_image)
    cv2.waitKey(0)

    return (crop, perturbed_crop, four_point_homography)

def perturb(x, y):
    return (x + random.randint(-perturbation_range, perturbation_range),
            y + random.randint(-perturbation_range, perturbation_range))

def prepare_dataset(dataset_path, destination_path):
    # Asumimos que son todas imagenes
    dataset_filenames = glob.glob(dataset_path + "*")

    for i, filename in enumerate(dataset_filenames):
        crop, perturbed_crop, homography = prepare_image_for_dataset(filename)
        cv2.imwrite(destination_path + str(i) + ".png", crop)
        cv2.imwrite(destination_path + str(i) + "_perturbed.png", perturbed_crop)

        with open(destination_path + str(i) + ".csv", mode='w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(homography)

def main():
    random.seed(1337)
    prepare_dataset("./resources/base_images/", "./resources/train_dataset/")

if __name__ == "__main__":
    main()