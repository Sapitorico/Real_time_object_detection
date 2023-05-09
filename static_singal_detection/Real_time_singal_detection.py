#!/usr/bin/python
import cv2
import mediapipe as mp
import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.saving import load_model

modelo = "CNN_Model/Modelo/Modelo.h5"
peso = "CNN_Model/pesos/pesos.h5"
direction = "mp_dataset/Train_Alphabet"
dire_img = os.listdir(direction)
cnn = load_model(modelo)  #Cargamos el modelo
cnn.load_weights(peso)  #Cargamos los pesos

import cv2
from Base_Model.Detection_tools import Base_Model

DATA_PATH = 'mp_dataset'
actions = []
imgSize = 200
size_data = 1000
save_frequency = 10
hand_type = "Right"
key = 0
count = 0

Base = Base_Model(DATA_PATH, actions, imgSize, size_data)
hands = Base.Hands_model_configuration(False, 1, 1)

predictions = []
threshold = 0.5

capture = cv2.VideoCapture(1)

with hands as Hands:
    while capture.isOpened():
        key = cv2.waitKey(1)
        success, image = capture.read()
        if not success:
            continue
        image = cv2.flip(image, 1)
        frame, results = Base.Hands_detection(image, Hands)
        copie_img = frame.copy()
        if results.multi_hand_landmarks:
            positions = []
            positions = Base.Detect_hand_type(hand_type, results, positions, copie_img)
            if len(positions) != 0:
                predicted_action = ""
                resized_hand = Base.Get_bound_boxes(positions, copie_img)
                img_array = img_to_array(resized_hand)
                result = cnn.predict(np.expand_dims(img_array, axis=0))[0]
                predictions.append(np.argmax(result))
                if np.unique(predictions[-10:])[0] == np.argmax(result):
                    if result[np.argmax(result)] > threshold:
                        predicted_action = dire_img[np.argmax(result)]
                Base.Draw_Bound_Boxes(positions, frame, predicted_action)
        if key == 27:
             exit(0)
        cv2.imshow("image capture", frame)
    capture.release()
    cv2.destroyAllWindows()