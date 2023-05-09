# Detección de lenguaje de señas para sordos usando Vision artificial

Debido a los distintos alfabetos de diferentes lenguajes, el lenguaje de se;as varía segun la region,
Intentare crear un intérprete de lenguajes de se;as USL.


## Primera fase del proyecto:
En esta primera parte consta de crear un intérprete de señas, que traduzca de lenguaje de señas a letras del alfabeto,

## Propuestas
La primera propuesta es crear una herramienta educativa, para fomentar el aprendizaje de lenguaje de señas, con clases pre grabadas, y challenge y actividades interactivas con la cámara.

La segunda propuesta se basa en que, muchas empresas optan por canales   de comunicación por video, como Zoom o Google Meet, entre otras.
Para organizar reuniones cooperativas, comunicación entre equipos de trabajo, entrevistas etc.
Este software tiene como objetivo brindar un sistema en tiempo real para el reconocimiento de los gestos de las manos.
Basándose en la detección de algunas características escanciale, como la orientación, el centro de gravedad, la posición espacial del cuerpo, el estado de los dedos. Teniendo en cuenta las similitudes de la forma de la mano humana.


# Boceto de la arquitectura



#  MediaPipe, pre procesar imágenes para crear puntos de referencia

Mediapie contiene una gran variedad de algoritmos o redes pre-entrenadas de detección y seguimiento del cuerpo humano, que se han entrenado en grandes cantidades de datos de Google.
Realizan un seguimiento de puntos críticos en diferentes secciones del cuerpo como el esqueleto de nodos y bordes o puntos de referencia. Todos los puntos de coordenadas se normalizan en tres dimensiones.