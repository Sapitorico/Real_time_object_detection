import cv2      # libreria de opencv
import mediapipe as mp      # libreria de redes de reconocimiento de articulaciones

mp_drawing = mp.solutions.drawing_utils     # funcion para dibujar los puntos y coneciones
mp_holistic = mp.solutions.holistic         # modelo pre-entrenado para reconocer articulaciones

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

# modelo mediapipe
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    smooth_landmarks=True
) as holistic:
    while capture.isOpened() and cv2.waitKey(1) & 0xff != 27:
        success, frame = capture.read()
        if not success:
            print("Ignorar camara vacia")
            continue
        # realizamos la deteccion
        image, results = mediapipe_detection(frame, holistic)
        print(results.left_hand_landmarks)

        # dibujamos los puntos de referencia
        draw_landmarks(image, results)
        # visualizar la imagen en modo espejo
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

    capture.release()
    cv2.destroyAllWindows()
