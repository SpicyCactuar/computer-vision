TODOs
=====

* Descargar dataset MS-COCO
* Generar dataset de entrenamiento
* Generar dataset de test

* Generar la red neuronal convolucional de bloques de 3x3 con Batch-Norm y ReLU:
    * Armar input de imagenes con 2 channels en greyscale
    * Armar 8 layers de 64, 64, 64, 64, 128, 128, 128, 128 filtros respectivamente
    * Agregar Dropout de 0.5 al final de la última layer convolucional
    * Armar 1 layer totalmente conectada de 1024 unidades con Droput de 0.5
    * Armar 1 layer Euclidean (L2) loss, de 8 unidades
* Armar pipeline de entrenamiento:
    * Stochastic Gradient Descent (SGD) con Momentum de 0.9
    * Base learning rate de 0.005 y decrecimiento del por un factor de 10 c/30.000 iteraciones
    * 90.000 iteraciones en total
    * Batch size de 64
    * los pesos de la red son inicializados al azar