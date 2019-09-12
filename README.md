# Pix2Pix-Deblurrer

Utilizar el modelo Pix2Pix para entrenar a una IA y corregir imagenes desenfocadas.

## Recursos:

| Link                      | Descripción
| :--------------           | :----------
| http://stylegan.xyz/paper | Más ejemplos.
| http://stylegan.xyz/video | Result video.
| http://stylegan.xyz/code  | Codigo original.
| http://stylegan.xyz/ffhq  | Flickr-Faces-HQ dataset.
| http://stylegan.xyz/drive | Google Drive folder.

## Requisitos:

Toda la red fue programada y entrenada desde un entorno de google Colab ejecutado de manera local en una Nvidia GTX 960 4Gb (debido a las limitaciones de tiempo de google Colab y a fallos de sincronización con google drive)

Es altamente recomendable utilizar una o varias GPUs Nvidia, intentamos crear una version del programa para ROCm y poder entrenar con una AMD RX 470 pero nos fue imposible.

- Ubuntu 18 o similares
- Tensorflow 2.0 rc0
- Numpy 1.18.0
- Pillow 6.1.0
- python 64-bit 3.7.3
- GPU Nvidia
- Jupyter notebook (para el entorno local de google Colab)
- Driver Nvidia 418.2 o mas reciente
- CUDA toolkit 9.0 o mas reciente
- cuDNN 7.3.1 o mas reciente



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
Cada iteracion tarda unos xx minutos en una GTX 960 con `n=1000` y fue entrenado durante 500 iteraciones (recomendable minimo 400 iteraciones para buenos resultados)







