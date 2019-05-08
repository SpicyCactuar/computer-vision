Deep Image Homography Estimation
=====

Implementación del paper [**Deep Image Homography Estimation**, de DeTone et al.](https://arxiv.org/abs/1606.03798)  
Para más información, leer `informe.pdf`.

### Estructura archivos
- `dataset_generator.py`: script para generar el dataset a partir del dataset de entrenamiento [MS-COCO 2014](http://cocodataset.org/#download).
- `clean_data.py`: script para limpiar imágenes que sean de sólo un color o estén falladas. No es necesario para el correcto entrenamiento de la red.  
- `homography_net_per_epoch_restarting_from_checkpoint.py`: script para entrenar la red neuronal de a una epoch a la vez, guardar un checkpoint y reanudar a partir del mismo si se corre otra vez.
- `homography_net.py`: script original que corría las 12 epochs todas seguidas. Debería funcionar, pero es recomendable usar el script anterior.
- `/reconstruccion`: carpeta con scripts para reconstruir una imagen a partir de la homografía estimada.
- `/fails`: carpeta que tiene imágenes con perturbaciones muy grandes, como ejemplo.
