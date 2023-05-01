import cv2      # libreria de opencv
import mediapipe as mp      # libreria de redes de reconocimiento de articulaciones
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils     # funcion para dibujar los puntos y coneciones
mp_holistic = mp.solutions.holistic         # modelo pre-entrenado para reconocer articulaciones


#-------------[ Rutas y tama;os de datos ]------------------------
# ruta para exportar los datos
DATA_PATH = os.path.join('mp_data')
# acciones a detectar
actions = np.array(['hola'])
# cantidad de videos por valor de dato
no_sequences = 30
# duracion de videos en cuadros frames
sequence_length = 30

#---[ Funcion para crear las rutas y carpetas ]-----------
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass



#---------------------[ Funciones ]-----------------------

#---- Funcion de preprocesado de la imagen y deteccion -----

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # convertimos el formato de colores
    image.flags.writeable = False                       # esto evita que no se modifique la imagen
    results = model.process(image)                      # procesamos la imagen con el modelo de detecion
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # reconvertimos la imagen a BGR
    return image, results                               # devolvemos la imagen procesada y el resultado de la deteccion

#---- Funcione de configuracion para vizualisar lso putnos de referencia y articulaciones ----

def draw_landmarks(image, results):
    # rostro:
    mp_drawing.draw_landmarks(
        image,                      # imagen preprocesada donde se dibujara los resputados del reconocimineto
        results.face_landmarks,     # conjutno de puntos que se detectaron en el rostro
        mp_holistic.FACEMESH_TESSELATION,   # puntos de referencia que se van a dbujar
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),  # color pixel de los puntos
        mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=1)    # color y pixel de las conceciones entre cada putno
    )
    # mano izquierda:
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,    # lista de puntos de referencia de la mano izquierda
        mp_holistic.HAND_CONNECTIONS,   # coneciones entre los puntos
        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )
    # mano derecha:
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,   # puntos de referencia de la mano derecha
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(53, 143, 0), thickness=2)
    )
    # postura:
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,     # puntos de refernecia de las poses del cuerpo
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
    )


#------------------[ Funciones de Procesamiento y Recopilamiento de Datos y Coordenadas ]----------------------------
"""
Cada punto de referencia de la pose se compone de cuatro valores: las coordenadas x, y, z y la
visibilidad del punto. Como los puntos de referencia de la pose son 33, el tamaño total de la matriz
"pose" es de 33 x 4 = 132.

El tamaño de la matriz face es 1404 porque se están tomando en cuenta los 468 puntos de referencia de
la cara, cada uno con tres coordenadas (x, y, z). Entonces, la matriz face resultante es de tamaño 468 x 3 = 1404.

El tamaño de 63 se debe a que hay 21 puntos de referencia y cada uno tiene 3 coordenadas
(x, y, z), lo que da un total de 63 valores en la matriz.

"""

def extract_keypoints(results):
    # Obtener los puntos de referencia de la pose y crear una matriz "pose" de tamaño 132
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        pose = np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten()
    else:
        pose = np.zeros(132)
    # Obtener los puntos de referencia faciales y crear una matriz "face" de tamaño 1404
    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        face = np.array([[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks]).flatten()
    else:
        face = np.zeros(1404)
    # Obtener los puntos de referencia de la mano izquierda y crear una matriz "lh" de tamaño 63
    if results.left_hand_landmarks:
        left_hand_landmarks = results.left_hand_landmarks.landmark
        lh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in left_hand_landmarks]).flatten()
    else:
        lh = np.zeros(21 * 3)
    # Obtener los puntos de referencia de la mano derecha y crear una matriz "rh" de tamaño 63
    if results.right_hand_landmarks:
        right_hand_landmarks = results.right_hand_landmarks.landmark
        rh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in right_hand_landmarks]).flatten()
    else:
        rh = np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)       # captura de video, CAP_DSHOW da mas seguridad

#----opciones de configuracion------
"""
static_image_mode: eso es para indicar si se va a procesar una
imagen fija o en una secuencua de imagenes(en tiempo real)
por defecto: False - se ejecuta en tiempo real

model_complexity: ajusta la complejidad del modelo de detecion,
cuanto mas complejo mejores resutlados, aun que mas latencia en la imagen
    por defecto: 1

smooth_landmarks: ajusta y suaviza la posicion de los puntos clave detectados
en la cara, mejor la estabilidad y la posicion de los puntos
    por defecto: True

min_detection_confidence: ajusta la confianza minima requerida para considerar
que un objeto ah sido detectado, osea que la confiansa de deteccion del modelo es
menor a la de MIN_DETECTION_CONFIDENCE, no se considerara el objeto
    por defecto: 0.5

min_tracking_confidence: ajusta el seguimiento del objeto, 
    por defecto: 0.5
"""
value = False
# modelo mediapipe
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    smooth_landmarks=True
) as holistic:

    for action in actions:
        # Reproducción en bucle de secuencias y vídeos
        for sequence in range(no_sequences):
            # Bucle a través de la duración del vídeo, también conocida como duración de la secuencia
            for frame_num in range(sequence_length):
                success, frame = capture.read()
                # realizamos la deteccion
                image, results = mediapipe_detection(frame, holistic)

                # dibujamos los puntos de referencia
                draw_landmarks(image, results)

                image = cv2.flip(image, 1)
                #  Aplicar lógica de espera
                if frame_num == 0:
                    cv2.putText(image, 'INICIO DE LA RECOGIDA', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image,
                                'Recopilacion de fotogramas para {} Numero de video {}'.format(action, sequence),
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Mostrar en pantalla
                    cv2.imshow('MediaPipe Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image,
                                'Recopilacion de fotogramas para {} Numero de video {}'.format(action, sequence),
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Mostrar en pantalla
                    cv2.imshow('MediaPipe Feed', image)

                # Exportar puntos clave
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == 13:
                break

    capture.release()
    cv2.destroyAllWindows()