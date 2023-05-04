import cv2      # libreria de opencv
import mediapipe as mp      # libreria de redes de reconocimiento de articulaciones
import os
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
#-------------------------[ Rutas y tamaños de datos ]--------------------------------
# definir la ruta donde se encuentran los datos
DATA_PATH = os.path.join('mp_data')
# definir las acciones a detectar
actions = np.array(['hola', 'mi', 'nombre es', 'Sapito'])
# cantidad de videos por valor de dato
no_sequences = 30
# duracion de videos en cuadros frames
sequence_length = 30
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(Dropout(0.2))     # 20% de las neuronas seran desactivadas
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('action.h5')


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (16, 117, 245)]

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

# def prob_viz(res, actions, image, colors, show_percentage=True):
#     for i, action in enumerate(actions):
#         # Draw bar
#         bar_x = 700
#         bar_y = 100 + i * 60
#         bar_width = int(res[i] * 400)
#         cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 40), colors[i], -1)
#
#         # Draw text
#         text = f"{action}"
#         if show_percentage:
#             text += f" ({int(res[i]*100)}%)"
#         cv2.putText(image, text, (bar_x, bar_y + 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#     return image
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    print(f"Prediction probability: {res[np.argmax(res)] * 100:.2f}%")

    return output_frame


# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_landmarks(image, results)

        # Check if at least one hand is present in the results
        if results.left_hand_landmarks or results.right_hand_landmarks:

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        predicted_action = actions[np.argmax(res)]
                        expected_action = "EXPECTED_ACTION"  # replace with the expected action

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                frame = prob_viz(res, actions, frame, colors)

            cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            sentence = []
            predictions = []

        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()