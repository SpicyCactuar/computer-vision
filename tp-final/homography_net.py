#!/usr/bin/env python
# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np
import cv2
import glob
from numpy import genfromtxt

checkpoint_path = "checkpoints/cp.ckpt"

#####################################################
# Datos
#####################################################
original_files = sorted(glob.glob("./resources/train_dataset/*_original.png"))
perturbed_files = sorted(glob.glob("./resources/train_dataset/*_perturbed.png"))

length = len(original_files)
max_train_index = int(length * 0.9)

original_train_files = original_files[0: max_train_index]
original_test_files = original_files[max_train_index:]

perturbed_train_files = original_files[0: max_train_index]
perturbed_test_files = perturbed_files[max_train_index:]

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
			#print(batch_original[i])
			#print(batch_perturbed[i])

			original = cv2.imread(batch_original[i], cv2.IMREAD_GRAYSCALE)
			perturbed = cv2.imread(batch_perturbed[i], cv2.IMREAD_GRAYSCALE)

			composite = np.zeros((128, 128, 2))
			composite[:,:,0] = original/255.0 #normalizar
			composite[:,:,1] = perturbed/255.0

			batch_x.append(composite)
			slash_index = batch_original[i].rfind('/')
			label = batch_original[i][0:slash_index] + batch_original[i][slash_index:batch_original[i].rfind('_')] + ".csv"
			#print(label)

			label_array = genfromtxt(label, delimiter=",")
			#print(label_array)
			batch_y.append(label_array)

		return np.array(batch_x), np.array(batch_y)

sequential_input = ImageSequence(original_train_files, perturbed_train_files, 64)

#####################################################
# Modelo
#####################################################
# Filtros de: 64, 64, 64, 64, 128, 128, 128, 128
model = tf.keras.models.Sequential([
	# imágenes de 128 x 128 y 2 canales grayscale.
	tf.keras.layers.Conv2D(input_shape=(128, 128, 2), data_format="channels_last", filters=64, kernel_size=(3, 3), strides=1, padding="same", use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'),
	# Batch Normalization puede hacerse antes ó después de la función de activación y la comunidad de machine learning está dividida en qué es mejor.
	# Aunque es default, el axis=-1 funciona para data_format="channels_last", si fuera "channels_first" sería axis=1
	tf.keras.layers.BatchNormalization(axis=-1),
	# La función de activación puede ponerse tanto en la capa de convolución como después.
	tf.keras.layers.Activation(tf.nn.relu),

	tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	# cada 2 hay max pooling
	tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"),

	tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	# cada 2 hay max pooling
	tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"),

	# Ahora vienen las de 128 filtros
	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	# cada 2 hay max pooling
	tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"),

	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	# Dropout de 0.5 después de la última layer convolucional
	tf.keras.layers.Dropout(0.5),

	# layer totalmente conectada
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1024, activation=tf.nn.relu),
	tf.keras.layers.Dropout(0.5),

	# última layer totalmente conectada, 8 unidades
	tf.keras.layers.Dense(8)
])
# Esto es para ver el modelo. Se puede comentar.
#model.summary()

def euclidean_distance_loss(y_true, y_pred):
	return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1))

# Compilar modelo con loss=l2, SGD con Momentum de 0.9, learning rate de 0.005
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
			  loss=euclidean_distance_loss,
			  metrics=['accuracy', 'mean_squared_error'])

#####################################################
# Entrenamiento
#####################################################

# Callback para el decay por un factor de 10 c/30.000 iteraciones
def step_decay(epoch):
	initial_lrate = 0.005
	epochs_drop = 4
	step = int(epoch/epochs_drop)
	if step==0:
		lrate = initial_lrate
	else:
		lrate = initial_lrate / float(10*step)
	return lrate

lRateSchedulerCallback = tf.keras.callbacks.LearningRateScheduler(step_decay)

# Para generar checkpoints y así no morir al entrenar.
# TODO: Hacer otro script que use este modelo y entrene a partir del último checkpoint.
checkpointCallback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, verbose=1)

# Ahora sí, a entrenar!
model.fit(sequential_input, epochs=12, batch_size=64, callbacks=[lRateSchedulerCallback, checkpointCallback], use_multiprocessing=True, workers=16)
# Ponerle que use multiprocessing y más workers no cambia nada ¯\_(ツ)_/¯

# Guardar el modelo entrenado.
model.save('modelo_regresion_entrenado.h5')

#####################################################
# Evaluación y predicción
#####################################################
test_sequence = ImageSequence(original_test_files, perturbed_test_files, 64)
# Evaluar el modelo:
print(model.evaluate(test_sequence, use_multiprocessing=True, workers=16))
