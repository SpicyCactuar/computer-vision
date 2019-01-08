#!/usr/bin/env python
# -*- coding: utf-8 -*
import tensorflow as tf
mnist = tf.keras.datasets.mnist # Esto no lo vamos a usar... Borrar cuando tengamos un dataset.

checkpoint_path = "checkpoints/cp.ckpt"

# TODO: Cargar los datos y pasárselos a la red.
# TODO: PROBAR que funcione todo lo que escribí corriéndolo una vez por lo menos :P
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Filtros de: 64, 64, 64, 64, 128, 128, 128, 128
model = tf.keras.models.Sequential([
	# imágenes de 128 x 128 y 2 canales grayscale. Necesario pasar el batch size acá?
	tf.keras.layers.Conv2D(input_shape=(64, 128, 128, 2), data_format="channels_last", filters=64, kernel_size=(3,3), strides=1, padding="valid", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	# Batch Normalization puede hacerse antes ó después de la función de activación y la comunidad de machine learning está dividida en qué es mejor.
	# Aunque es default, el axis=-1 funciona para data_format="channels_last", si fuera "channels_first" sería axis=1
	tf.keras.layers.BatchNormalization(axis=-1),
	# La función de activación puede ponerse tanto en la capa de convolución como después.
	tf.keras.layers.Activation(tf.nn.relu),

	tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="valid", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	# cada 2 hay max pooling
	tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="valid"),

	tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="valid", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="valid", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	# cada 2 hay max pooling
	tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="valid"),

	# Ahora vienen las de 128 filtros
	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="valid", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="valid", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	# cada 2 hay max pooling
	tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="valid"),

	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="valid", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding="valid", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
	tf.keras.layers.BatchNormalization(axis=-1),
	tf.keras.layers.Activation(tf.nn.relu),

	# Dropout de 0.5 después de la última layer convolucional
	tf.keras.layers.Dropout(0.5),

	# layer totalmente conectada
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1024, activation=tf.nn.relu),
	tf.keras.layers.Dropout(0.5),

	# última layer totalmente conectada, 8 unidades, softmax
	tf.keras.layers.Dense(8, activation=tf.nn.softmax)
])

# Compilar modelo con loss=l2, SGD con Momentum de 0.9, learning rate de 0.005
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
			  loss=tf.nn.l2_loss,
			  metrics=['accuracy'])

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

lRateSchedulerCallback = LearningRateScheduler(step_decay)

# Para generar checkpoints y así no morir al entrenar.
# TODO: Hacer otro script que use este modelo y entrene a partir del último checkpoint.
checkpointCallback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, verbose=1)

# Ahora sí, a entrenar!
model.fit(x_train, y_train, epochs=12, batch_size=64, callbacks=[lRateSchedulerCallback, checkpointCallback])

# Guardar el modelo entrenado.
model.save('modelo_regresion_entrenado.h5')

# Evaluar el modelo:
#model.evaluate(x_test, y_test)
