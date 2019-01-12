#!/usr/bin/env python
# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np
import cv2

checkpoint_path = "checkpoints/cp.ckpt"

#####################################################
# Datos
#####################################################

img = cv2.imread("Paulo.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("Paulo2.png", cv2.IMREAD_GRAYSCALE)

composite = np.zeros((128, 128, 2))
composite[:,:,0] = img/255.0 #normalizar
composite[:,:,1] = img2/255.0

labels = [-30.0, -20.0, 20.0, -12.0, -16.0, 41.0, 25.0, 30.0]

x_train = np.array([composite] * 1024)
y_train = np.array([labels] * 1024)

# TODO: Hacer una clase secuencia que CARGUE las imágenes de a batches y las preprocese así no tenemos 500000 imágenes en memoria. Esto es sólo para probar multithreading. Ignorar.
class ImageSequence(tf.keras.utils.Sequence):
	def __init__(self, x_set, y_set, batch_size):
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size

	def __len__(self):
		return int(np.ceil(len(self.x) / float(self.batch_size)))

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

		return batch_x, batch_y

sequentialInput = ImageSequence(x_train, y_train, 64)

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
			  metrics=['accuracy'])

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
model.fit(x_train, y_train, epochs=12, batch_size=64, callbacks=[lRateSchedulerCallback, checkpointCallback])
#model.fit(sequentialInput, epochs=12, batch_size=64, callbacks=[lRateSchedulerCallback, checkpointCallback], use_multiprocessing=True, workers=16)
# Ponerle que use multiprocessing y más workers no cambia nada ¯\_(ツ)_/¯

# Guardar el modelo entrenado.
model.save('modelo_regresion_entrenado.h5')

#####################################################
# Evaluación y predicción
#####################################################

print(model.predict(np.array([composite])))

composite2 = np.zeros((128, 128, 2))
composite2[:,:,0] = img/255.0 #normalizar
composite2[:,:,1] = img/255.0

print(model.predict(np.array([composite2])))

# Evaluar el modelo:
#model.evaluate(x_test, y_test)
