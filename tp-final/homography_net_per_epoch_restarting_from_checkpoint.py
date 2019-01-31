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

batch_size = 64

#####################################################
# Datos
#####################################################
original_files = sorted(glob.glob("./resources/train_dataset/*_original.png"))
perturbed_files = sorted(glob.glob("./resources/train_dataset/*_perturbed.png"))

length = len(original_files)
max_train_index = int(length * 0.9)

original_train_files = original_files[0: max_train_index]
original_test_files = original_files[max_train_index:]

perturbed_train_files = perturbed_files[0: max_train_index]
perturbed_test_files = perturbed_files[max_train_index:]

# Randomizar el orden de los datos de train, para que no haya bias por orden.
combined = list(zip(original_train_files, perturbed_train_files))
random.shuffle(combined)

original_train_files[:], perturbed_train_files[:] = zip(*combined)

class ImageSequence(tf.keras.utils.Sequence):
	def __init__(self, original, perturbed, batch_size):
		self.original_files = original
		self.perturbed_files = perturbed
		self.batch_size = batch_size

	def __len__(self):
		return int(np.ceil(len(self.original_files) / float(self.batch_size)))

	# Dar un batch de imágenes.
	def __getitem__(self, idx):
		batch_original = self.original_files[idx * self.batch_size : (idx + 1) * self.batch_size]
		batch_perturbed = self.perturbed_files[idx * self.batch_size : (idx + 1) * self.batch_size]

		batch_x = []
		batch_y = []
		for i in range(len(batch_original)):

			original = cv2.imread(batch_original[i], cv2.IMREAD_GRAYSCALE)
			perturbed = cv2.imread(batch_perturbed[i], cv2.IMREAD_GRAYSCALE)

			composite = np.zeros((128, 128, 2))
			composite[:,:,0] = (original - 127.5) / 127.5 #normalizar
			composite[:,:,1] = (perturbed - 127.5) / 127.5

			batch_x.append(composite)
			label = batch_original[i].replace("_original.png", ".csv")
			#print("original: " + batch_original[i] + "\nperturbed: " + batch_perturbed[i] + "\nlabel: " + label)

			label_array = genfromtxt(label, delimiter=",")
			#print(label_array)

			batch_y.append(label_array / 16.0)

		return np.array(batch_x), np.array(batch_y)

sequential_input = ImageSequence(original_train_files, perturbed_train_files, batch_size)

#####################################################
# Modelo
#####################################################

def euclidean_distance_loss(y_true, y_pred):
	return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1, keepdims=True))

# Esto encontrado en internet
def mace(y_true, y_pred):
    return tf.keras.backend.mean(16*tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(tf.keras.backend.reshape(y_pred, (-1,4,2)) - tf.keras.backend.reshape(y_true, (-1,4,2))), axis=-1, keepdims=True)),axis=1)

checkpoint_files = glob.glob("checkpoint_model_*")
checkpoint_path = ""

def extract_number(f):
	s = re.search("[0-9]+",f)
	s = s.group(0)
	return (int(s) if s else -1,f)

if len(checkpoint_files) == 0:
	# Filtros de: 64, 64, 64, 64, 128, 128, 128, 128
	model = tf.keras.models.Sequential([
		# imágenes de 128 x 128 y 2 canales grayscale.
		tf.keras.layers.Conv2D(input_shape=(128, 128, 2), data_format="channels_last", filters=64, kernel_size=3, padding="same", activation="relu"),
		tf.keras.layers.BatchNormalization(),

		tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
		tf.keras.layers.BatchNormalization(),

		tf.keras.layers.MaxPool2D(pool_size=2),

		tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
		tf.keras.layers.BatchNormalization(),

		tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
		tf.keras.layers.BatchNormalization(),

		tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same"),

		tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
		tf.keras.layers.BatchNormalization(),

		tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
		tf.keras.layers.BatchNormalization(),

		tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same"),

		tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
		tf.keras.layers.BatchNormalization(),

		tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
		tf.keras.layers.BatchNormalization(),

		tf.keras.layers.Flatten(),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(1024, activation="relu"),

		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(8)
	])

	# Necesito compilar sólo si no tenía compilado el modelo antes!
	# Compilar modelo con loss=l2, SGD con Momentum de 0.9, learning rate de 0.005
	model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
				  loss=euclidean_distance_loss,
				  metrics=['accuracy', 'mean_squared_error', mace])

	print("Starting training")
else:
	checkpoint_path = max(checkpoint_files, key=extract_number)

	model = tf.keras.models.load_model(checkpoint_path, custom_objects={'euclidean_distance_loss': euclidean_distance_loss, 'mace': mace})
	print("Training from: " + checkpoint_path)


# Esto es para ver el modelo. Se puede comentar.
#model.summary()

#####################################################
# Entrenamiento
#####################################################

# Callback para el decay por un factor de 10 c/30.000 iteraciones
# def step_decay(epoch):
# 	initial_lrate = 0.00005
# 	epochs_drop = 4
# 	step = int(epoch/epochs_drop)
# 	if step==0:
# 		lrate = initial_lrate
# 	else:
# 		lrate = initial_lrate / float(10*step)
# 	return lrate
#
# lRateSchedulerCallback = tf.keras.callbacks.LearningRateScheduler(step_decay)

# Ahora sí, a entrenar!
model.fit(sequential_input, epochs=1, batch_size=batch_size, callbacks=[], use_multiprocessing=True, workers=16)
# Ponerle que use multiprocessing y más workers no cambia nada ¯\_(ツ)_/¯

# Guardar el modelo entrenado.
if len(checkpoint_files) > 0:
	i = extract_number(checkpoint_path)[0] + 1
else:
	i = 0
model.save("checkpoint_model_%s.h5" % i)

#####################################################
# Evaluación y predicción
#####################################################
#print("Testing...")
#test_sequence = ImageSequence(original_test_files, perturbed_test_files, 64)
# Evaluar el modelo:
#print(model.evaluate(test_sequence, use_multiprocessing=True, workers=16))
#print(model.predict(test_sequence))
