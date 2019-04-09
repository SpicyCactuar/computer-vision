#!/usr/bin/env python
# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np
import cv2
import glob
from numpy import genfromtxt
import os.path
import re
import random

#####################################################
# Modelo
#####################################################

def euclidean_distance_loss(y_true, y_pred):
	return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1, keepdims=True))

# Esto encontrado en internet
def mace(y_true, y_pred):
    return tf.keras.backend.mean(16*tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(tf.keras.backend.reshape(y_pred, (-1,4,2)) - tf.keras.backend.reshape(y_true, (-1,4,2))), axis=-1, keepdims=True)),axis=1)

checkpoint_path = "checkpoint_model_14.h5"

model = tf.keras.models.load_model(checkpoint_path, custom_objects={'euclidean_distance_loss': euclidean_distance_loss, 'mace': mace})

#####################################################
# Predicción
#####################################################

original_image = cv2.imread("cat_original.png", cv2.IMREAD_GRAYSCALE)
perturbed_image = cv2.imread("cat_perturbed.png", cv2.IMREAD_GRAYSCALE)

composite = np.zeros((128, 128, 2))
composite[:,:,0] = (original_image - 127.5) / 127.5 #normalizar
composite[:,:,1] = (perturbed_image - 127.5) / 127.5

data = np.zeros((1, 128, 128, 2))
data[0] = composite

original_homography = genfromtxt("cat.csv", delimiter=",")
print("Original homography: " + str(original_homography))

predicted_homography = model.predict(data) * 16

print("Predicted homography: " + str(predicted_homography))

#####################################################
# Creación de nueva imagen
#####################################################

pX = 287
pY = 114

original_points = np.float32([
	(pX, pY),
	(pX + 128, pY),
	(pX + 128, pY + 128),
	(pX, pY + 128)
])

predicted_homography = predicted_homography.flatten().reshape((4,2))
#predicted_homography[:,0] = predicted_homography[:,0]/128.0 * 640.0
#predicted_homography[:,1] = predicted_homography[:,1]/128.0 * 480.0

perturbed_points = original_points + predicted_homography

big_image = cv2.imread("cat_1.png", cv2.IMREAD_GRAYSCALE)
big_image = cv2.resize(big_image, (640, 480))

M = cv2.getPerspectiveTransform(original_points, perturbed_points)
new_image = cv2.warpPerspective(big_image, M, (640, 480), flags=cv2.WARP_INVERSE_MAP)
cv2.imwrite("new_cat.png", new_image[114:114+128,287:287+128])
