# Pix2Pix-Deblurrer

Utilizar el modelo Pix2Pix para entrenar a una IA y corregir imagenes desenfocadas.

## Recursos:

| Link                      | Descripción
| :--------------           | :----------
| https://arxiv.org/pdf/1611.07004.pdf| Paper original Pix2Pix.
| http://stylegan.xyz/video | Mas ejemplos.
| http://stylegan.xyz/code  | Codigo original.
| https://github.com/NVlabs/ffhq-dataset  | Flickr-Faces-HQ dataset.
| http://stylegan.xyz/drive | Carpeta Google Drive.

## Requisitos:

Toda la red fue programada y entrenada desde un entorno de google Colab ejecutado de manera local en una Nvidia GTX 960 4Gb (debido a las limitaciones de tiempo de google Colab y a fallos de sincronización con google drive)

Es altamente recomendable utilizar una o varias GPUs Nvidia debido a la compatibilidad que proporciona tensorflow a las tarjetas Nvidia, existe una libreria llamada ROCm para ejectuar tensorflow en GPUs AMD pero carece de la cantidad de desarrollo y soporte que tiene CUDA. 

Hemos creado una version de Pix2Pix-Deblurrer para GPUs AMD, para utilizarla debes instalar ROCm suiguiendo [estos pasos](www)

- Ubuntu 18 o similares
- Tensorflow 2.0 rc0
- Numpy 1.18.0
- Pillow 6.1.0
- python 64-bit 3.7.3
- GPU Nvidia
- Jupyter notebook (para el entorno local de google Colab)

Nvidia:

- Driver Nvidia 418.2 o mas reciente
- CUDA toolkit 9.0 o mas reciente
- cuDNN 7.3.1 o mas reciente

Amd:

- ROCm



## Donde conseguir las imagenes:

Nosotros utilizamos el Flickr-Faces-HQ dataset, lo puedes en econtrar aqui [Flickr-Faces-HQ repository](https://github.com/NVlabs/ffhq-dataset).
Es altamente recomendable descargar las imagenes mediante el archivo de python que se encuentra en su repositorio


## Preprocesamiento de imagen:

Adobe Photoshop CC fue usado para modificar las imagenes del dataset FFHQ.

Para recrear nuestro dataset:
- Crear una macro en photoshop que desenfoque las imagenes originales (aqui puedes elegir el tipo de desenfoque y su intensidad,     recomendamos el desenfoque de lente ya que es lo más parecido a una imagen desenfocada de forma natural por una camara)
- Usar esa misma macro para reescalar todas las imagenes a 256 x 256 pixeles y guardlas en formato .jpg (importante que sea .jpg y   no cualquier otro)
- Exportar a una carpeta (en nuestro caso "/Blurred_resized")
- Repetir todo pero sin desenfocar y exportar a otra carpeta (en nuestro caso "/not_blurred_resized")



## Antes de entrenar:

Es necesario modificar ciertas variables en `run.py`

- Establecer `PATH` a ser la ruta completa hasta la carpeta donde se ubica `run.py`

- Establecer la variable `n` al numero de imagenes que deseas cargar para el entrenamiento y testeo (en un ratio 8:2)
  los resultados mostrados aqui han sido entrenados con `n = 1000` debido a nuestras limitaciones de GPU, en un sistema con
  multiples RTX 2080ti o Tesla V100 se podria configurar hasta `n = 71000` para usar todas las imagenes del dataset (ten en cuenta
  que eso incurre un costo computacional inmenso y por lo tanto tardará mucho más en completar su entrenamiento, a cambio generará 
  resultados mucho mas realistas y generalizables)
  

## El entrenamiento:

El entrenamiento emplea un `batch_size` de una sola imagen para maximizar fidelidad visual
Tensorflow automaticamente asigna recursos de hardware así que en principio no es necesario seleccionar si quieres entrenar con GPU o no, si esta disponible se usará.
Cada iteracion tarda unos 6 minutos en una GTX 960 con `n=1000` y fue entrenado durante 500 iteraciones (recomendable minimo 400 iteraciones para buenos resultados)


## Evaluación y guardado

El modelo trata de mejorar las imagenes generadas mediante un valor de error determinado por `total_gen_loss = gan_loss + (100 * pixel_loss)` donde `gan_loss` es el valor de error generado por el discriminador (patchGAN) y `pixel_loss` es la diferencia media en el valor de cada pixel entre la imagen original y la generada. `pixel_loss`esta multiplicado por `100`para que sea 100 veces mas significativo que el valor del discriminador lo cual mejora los resultados según el equipo de Pix2Pix detalla en su paper (link arriba).

Todos los pesos de tanto el generador como el discriminador asi como todas las variables son guardadas en la carpeta tf_checkpoints y recupera el ultimo guardado al ejecutar el codigo.






