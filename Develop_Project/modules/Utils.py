#!/usr/bin/python
import cv2
import mediapipe as mp  # libreria de redes de reconocimiento de articulaciones
import os
import numpy as np

# -------[ Modelo de reconocimiento de articulaciones ]------------------------------
#inicilisamos el modelo de detecion de manos
mp_hands = mp.solutions.hands
#funcion para visulaizar el resultado en puntos de referencia
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class Utils:
    def __init__(self, DATA_PATH=None, actions=None, imgSize=None, size_data=None):
        """
        construct utils

        :param DATA_PATH: dataset directory
        :param actions: class actions
        :param imgSize: size of image
        :param size_data: number of image

        """
        self.DATA_PATH=DATA_PATH
        self.actions=actions
        self.imgSize=imgSize
        self.size_data=size_data
        self.save_frequency=10
        self.offset=20

    @staticmethod
    def Hands_model_configuration(image_mode, max_hand, complexity):
        """
        configue mediapipe detection hand

        :param image_mode: detection mode
        :param max_hand: number of hands detected
        :param complexity: complexity of the model

        return: hand detection model
        """
        hands = mp_hands.Hands(
            static_image_mode=image_mode,
            max_num_hands=max_hand,
            model_complexity=complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return hands

    @staticmethod
    def Hands_detection(image, model):
        """
        image processing and landmark detection

        :param image: image
        :param model: hand detection model

        return: image and result
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = model.process(image)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, result

    @staticmethod
    def Detect_hand_type(hand_type, results, positions, copie_img):
        """
        tracking of the position of the hand in the image, depending on the image chosen by the user

        :param hand_type: Left or Right hand type
        :param results: detection results
        :param positions: list of positions
        :param copie_img: copy image

        return: position of the hand
        """
        for hand_index, hand_info in enumerate(results.multi_handedness):
            hand_types = hand_info.classification[0].label
            if hand_types == hand_type:
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        alto, ancho, c = copie_img.shape
                        positions.append((lm.x * ancho, lm.y * alto, lm.z * ancho))
        return positions

    def Draw_Bound_Boxes(self, positions, frame):
        """
        hand box

        :param positions: Position array
        :param frame: Frame
        """
        x_min = int(min(positions, key=lambda x: x[0])[0])
        y_min = int(min(positions, key=lambda x: x[1])[1])
        x_max = int(max(positions, key=lambda x: x[0])[0])
        y_max = int(max(positions, key=lambda x: x[1])[1])
        width = x_max - x_min
        height = y_max - y_min
        x1, y1 = x_min, y_min
        x2, y2 = x_min + width, y_min + height
        if y1 - self.offset - 25 >= 0 and y2 - self.offset + 50 <= frame.shape[
            0] and x1 - self.offset - 50 >= 0 and x2 - self.offset + 90 <= \
                frame.shape[1]:
            cv2.rectangle(frame, (x1 - self.offset - 50, y1 - self.offset - 25), (x2 - self.offset + 90, y2 - self.offset + 50),
                          (0, 255, 0), 3)

    def Get_image_resized(self, positions, copie_img):
        """
        image reshaping

        :param positions: x y z coordinates
        :param copie_img: copy frame

        return: redeemed image
        """
        alto, ancho, c = copie_img.shape
        x_min = int(min(positions, key=lambda x: x[0])[0])
        y_min = int(min(positions, key=lambda x: x[1])[1])
        x_max = int(max(positions, key=lambda x: x[0])[0])
        y_max = int(max(positions, key=lambda x: x[1])[1])
        width = x_max - x_min
        height = y_max - y_min
        centro_x, centro_y = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
        # Calcular las coordenadas del cuadro centrado en la mano
        lado = max(width, height)
        x1 = max(0, centro_x - int(lado / 2) - 50)
        y1 = max(0, centro_y - int(lado / 2) - 50)
        ancho = min(ancho - x1, int(lado) + 100)
        alto = min(alto - y1, int(lado) + 100)
        x2, y2 = x1 + ancho, y1 + alto
        resized_hand = copie_img[y1:y2, x1:x2]
        resized_hand = cv2.resize(resized_hand, (self.imgSize, self.imgSize), interpolation=cv2.INTER_CUBIC)
        return resized_hand

    def Create_datasets_dir(self):
        """
        creates the dataset directories
        """
        for action in self.actions:
            try:
                os.makedirs(os.path.join(self.DATA_PATH, action))
            except:
                pass

    def Save_resized_hand(self, resized_hand, count, hand_type):
        """
        saves the captured image

        :param resized_hand: image
        :param count: cont
        :param hand_type: Left or Right
        """
        for action in self.actions:
            for sequence in range(self.save_frequency):
                image_path = os.path.join(self.DATA_PATH, action, f'sequence {sequence} capture {hand_type} {count}.png')
                cv2.imwrite(image_path, resized_hand)
        if count >= self.size_data / self.save_frequency:
            exit(0)