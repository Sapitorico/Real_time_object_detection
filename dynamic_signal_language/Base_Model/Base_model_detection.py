#!/usr/bin/python
import mediapipe as mp      # libreria de redes de reconocimiento de articulaciones
import os
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils     # funcion para dibujar los puntos y coneciones
mp_holistic = mp.solutions.holistic         # modelo pre-entrenado para reconocer articulaciones
"""
class Detection:
    
"""

class Base_Model_Detection:
    def __init__(self, complex=1, min_detect=0.5, min_track=0.5):
        self.holistic_model = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=complex,
                min_detection_confidence=min_detect,
                min_tracking_confidence=min_track,
                smooth_landmarks=True
        )

    def Rutes_and_Sizes(self, data_path="", classes=[], sequences=30, frames=30):
        self.DATA_PATH = os.path.join(data_path)
        self.actions = np.array(classes)
        self.no_sequences = sequences
        self.sequence_length = frames

    def Create_Rutes(self, data_path="", classes=[], sequences=30):
        self.Rutes_and_Sizes(data_path, classes, sequences)
        for action in classes:
            for sequence in range(sequences):
                try:
                    os.makedirs(os.path.join(self.DATA_PATH, action, str(sequence)))
                except:
                    pass

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convertimos el formato de colores
        image.flags.writeable = False  # esto evita que no se modifique la imagen
        results = model.process(image)  # procesamos la imagen con el modelo de detecion
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # reconvertimos la imagen a BGR
        return image, results  # devolvemos la imagen procesada y el resultado de la deteccion

    def extract_keypoints(self, results):
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

    def draw_landmarks(self, image, results):
        # rostro:
        mp_drawing.draw_landmarks(
            image,  # imagen preprocesada donde se dibujara los resputados del reconocimineto
            results.face_landmarks,  # conjutno de puntos que se detectaron en el rostro
            mp_holistic.FACEMESH_TESSELATION,  # puntos de referencia que se van a dbujar
            mp_drawing.DrawingSpec(color=(255, 204, 204), thickness=1, circle_radius=1),  # color pixel de los puntos
            mp_drawing.DrawingSpec(color=(204, 102, 102), thickness=1))  # color y pixel de las conceciones entre cada putno
        # mano izquierda:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,  # lista de puntos de referencia de la mano izquierda
            mp_holistic.HAND_CONNECTIONS,  # coneciones entre los puntos
            mp_drawing.DrawingSpec(color=(204, 204, 0), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=2))
        # mano derecha:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,  # puntos de referencia de la mano derecha
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(102, 204, 102), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 102, 0), thickness=2))
        # postura:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,  # puntos de refernecia de las poses del cuerpo
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(204, 153, 255), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))


    def draw_text(self, frame_num, image, action, sequence):
        if frame_num == 0:
            cv2.putText(image, 'RECOLECTANO DATOS', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(image, 'Recopilacion de fotogramas para', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, '{}'.format(action), (95 * 3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(image, 'Numero de video', (109 * 3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, '{}'.format(sequence), (155 * 3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    cv2.LINE_AA)

