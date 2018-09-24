import cv2
import numpy as np
import math
import matplotlib.pyplot as plot

## Ejercicio 1a

def sum_images(img1, img2):
    return img1 + img2

def subtract_images(img1, img2):
    return img1 - img2

def multiply_images(img1, img2):
    return img1 * img2

## Ejercicio 1b

def scalar_multiply_image(img, scalar):
    return (scalar * img).astype(np.uint8)

## Ejercicio 1c

def dynamic_range_compression(img):
    L = 256
    R = img.max()
    constant = (L - 1) / math.log10(R + 1)
    transformed_img = np.log10(img + 1)
    return scalar_multiply_image(transformed_img, constant)

## Ejercicio 2

def negative_image(img):
    return 255 - img

## Ejercicio 3

def threshold_image(img, threshold):
    result = img.copy()
    result[result <= threshold] = 0
    result[result > threshold] = 255
    return result

## Ejercicio 4

def bit_level_splice(img):
    bit1 = img & 1
    bit2 = img & 2
    bit3 = img & 4
    bit4 = img & 8
    bit5 = img & 16
    bit6 = img & 32
    bit7 = img & 64
    bit8 = img & 128
    return bit1, bit2, bit3, bit4, bit5, bit6, bit7, bit8

## Ejercicio 5

def greyscale_histogram(img):
    counts, bins, bars = plot.hist(img.ravel(), bins=256)
    print(bins.astype(np.uint8))
    plot.show()

## Ejercicio 6

def contrast_image(img):
    contraster = lambda pixel: pixel * 0.5 if pixel <= 80 else (pixel if pixel <= 180 else min(pixel * 1.2, 255))
    return np.vectorize(contraster)(img).astype(np.uint8)

## Ejercicio 7

def equalize_image(img):
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf = cdf / float(cdf.max())

    minimum = cdf.min()
    result = ((cdf - minimum) * 255) / (1 - minimum) + 0.5
    result = result.astype(np.uint8)
    return result[img]

## Ejercicio 8

def double_equalize_image(img):
    return equalize_image(equalize_image(img))

## Explicación: La imagen no cambia, en relación a la primera equalización, ya que se intenta distribuir uniformemente algo que ya estaba distribuido uniformemente.

## Ejercicio 9 - WIP

def main():
    img1 = cv2.imread('greyscale_images/lena.png')
    img2 = cv2.imread('color_images/cat_1.png')

    result = double_equalize_image(img1)
    #result = contrast_image(img1)
    greyscale_histogram(result)
    cv2.namedWindow("window")
    #for img in list(result):
    cv2.imshow("window", result)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()