# Reconocimiento Alfabético usando Redes Neuronales Convoluciones (CNN)

Las redes convoluciones con excelentes para datos de imágenes, clasificación y reconocimiento.

La tarea es crear una red neuronal que aprenda a reconocer signos de manos y las intérprete como letras del alfabeto.
Are una arquitectura en TensorFlow para construir estas redes y entrenarlas con Google Colab,
tomaré la que tenga mejor resultado.

Estructura del proyecto:

-1: Recopilar datos, para las clases

-2: visualización de los datos

-3: pre-procesar datos y dividirlos

-4: preparar nuestro modelo para el aprendizaje

-5: entrenar el modelo

-6: Comprobar gráficos de prueba de precisión y perdida y guardar el modelo

-7: realizar pruebas en tiempo real

-8: crear una aplicación de escritorio para aplicar el modelo

Como va a funcionar:
La función básica es simple, en tiempo real de video, se hará un reconocimiento de las manos, podrás elegir la mano predominante.
Para el reconocimiento de la mano utilizaré mediapipe, resaltaré la mano con un recuadro que se redimensione según la posición espacial del la mano, x, y, z, esto para identificar la mano que se clasificara y la mano predominante.
Tomaré la imagen en un recuadro con un tamaño fijo, esta será la imagen que procesar la red neuronal en tiempo real, esto el usuario no lo verá.

## Proceso de visión
El proceso de visón, se basa en reconocer características de los objetos en ellas, para poder identificar, se basa en escanear la imagen y tomar todas estas características para clasificar y predecir que es lo que hay en la imagen,
Como sabrá el modelo a que clasificación pertenece una imagen, básicamente identificara una serie de elementos que conformen un cierta clase, identificara en un mapa de características, patrones de líneas, cambios de contraste, patrones circulares, etc.
Se basa en un procesamiento en cascada donde primero se identifican elementos básicos y generales, donde en posteriores capas se combinan para generar patrones más complejos.

## Clasificación de imágenes
la clasificación de la imagen será en tiempo real, obtendremos una predicción de clase mediante un cuadro-frame
* Estructura espacial:
    una imagen está compuesta por píxeles con valores de graduación, el valor de un píxel esta ligado al valor de los píxeles vecinos, tanto en ancho como en altura,
    esto es lo que hace que surjan estructuras, formas y patrones.

## CNN
Una red convolución es un tipo de red que aplica un tipo de capa que realiza una serie de operaciones matemáticas llamada convolución, sobre los píxeles , que generara una nueva imagen conocida como mapa de características.
Cada píxel nuevo se genera colocando una matriz de números llamado filtro o kernel sobre la imagen original, donde multiplicaremos y sumaremos cada valor de píxel vecino para obtener un nuevo valor,
si desplazamos este filtro realizando la convolución sobre toda la imagen, obtendremos una nueva imagen que será el mapa de características.
Esta operación de convolución puede detectar ciertas características o otras dependiendo de los valores del filtro, estos valores los aprenderá sola el modelo.
Esto es secuencial, la salida de una de las capas es la entrada de la siguiente, una cosa, cuando tenesmo una entrada de una imagen a color, como entrada haría 3 canales de color RGB donde cada uno de los canales es un mapa distinto de características.

## Enfoque 1: Recopilación de datos:
Aquí voy a crear una función para recopilar datos de y clasificarlos por clase,
para esto realizaremos la detección de la mano con un modelo pre-entrenado y luego aremos un recuadro de la imagen que se autoajuste dependiendo de la expansión de los dedos,
luego especificare una serie de rutas y nombres de clases, 
por último registraremos una cantidad de imágenes y guardarlas en un directorio de mp_dataset donde estarán clasificadas por directorios con los nombres de sus respectivas clases.
Las funciones estarán en una clase Base_Model, ordenadas, funciones para crear las rutas, preprocesamiento de las imágenes, etiquetado, guardado, etc

Librerias:

import cv2

import mediapipe as pm

import os

## Enfoque 2: Visualizar los datos
Creare una función que elija de forma aleatoria unas 10 imágenes de cada clase y las muestre con pyplot

Librerias:

import matplotlib.pyplot as plt

import os

import random

import cv2

## Enfoque 3: reprocesar datos y dividirlos
Cambiaremos todas las imágenes y etiquetas en una sola lista y luego las procesaremos según lo que requiera la red neuronal.
Cree una clase que contendrá todas las funciones de pre-procesado de imágenes, y el constructor del modelo de red convolución.
Tenemos funciones que crean los paquetes de entrenamiento y validación, una función que construye el modelo otra para entrenarla con el conjunto anteriormente creado y la última para guardar el modelo y sus pesos

Librerías:

from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import TensorBoard

import os

## Enfoque 4 Creacion del modelo
Preparamos nuestro modelo para el entrenamiento, hice la clase constructora y el archivo donde contiene las variables de dependencias y llamado a las respectivas funciones de creación y entrenamiento del modelo

## Enfoque 5 Entrenamiento del modelo
Estoy en proceso de entrenamiento del modelo con una dataset de 5 clases, iré agregando clases y modificando la estructura de la red, dependido de los resultados que obtenga
El entrenamiento lo aré en una libreta de Google Colab