#!/usr/bin/python
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.models import model_from_json
import numpy as np
import cv2



class LSTM_Model:
    json_file = open("./Models/model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # actions = np.array(
    #     ['Como estas', 'bien', 'de nada', 'encantado', 'gracias', 'hola', 'mal', 'mas o menos', 'muchas gracias',
    #      'nos vemos', 'vos'])
    # model = Sequential()
    # model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    # # model.add(Dropout(0.2))  # 20% de las neuronas seran desactivadas
    # model.add(LSTM(128, return_sequences=True, activation='relu'))
    # # model.add(Dropout(0.2))
    # model.add(LSTM(64, return_sequences=False, activation='relu'))
    # # model.add(Dropout(0.2))
    # model.add(Dense(64, activation='relu'))
    # # model.add(Dropout(0.2))
    # model.add(Dense(32, activation='relu'))
    # # model.add(Dropout(0.2))
    # model.add(Dense(actions.shape[0], activation='softmax'))
    model.load_weights('Models/Dinamic_model/action.h5')


    @staticmethod
    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
        print(f"Prediction probability: {res[np.argmax(res)] * 100:.2f}%")
        return output_frame