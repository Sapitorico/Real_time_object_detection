#!/usr/bin/python
import cv2
from Base_Model.Base_model_detection import Base_Model
capture = cv2.VideoCapture(2, cv2.CAP_DSHOW)

detect = Base_Model(complex=2)
with detect.holistic_model as holistic:
    while capture.isOpened() and cv2.waitKey(1) & 0xff != 27:
        success, frame = capture.read()
        if not success:
            print("Ignorar camara vacia")
            continue
        # realizamos la deteccion
        image, results = detect.mediapipe_detection(frame, holistic)
        # print(results.left_hand_landmarks)

        # dibujamos los puntos de referencia
        detect.draw_landmarks(image, results)
        # visualizar la imagen en modo espejo
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

    capture.release()
    cv2.destroyAllWindows()