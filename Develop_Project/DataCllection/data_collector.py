#!/usr/bin/python
import cv2
from Develop_Project.modules.Utils import Utils


# ruta de dataset
DATA_PATH = 'DATASET/Training'
# especifica la letra
actions = ['']
# tama;o d ela imagen
imgSize = 224
# cantidad de datos
size_data = 100
# Left Right
hand_type = "Left"
#key de boton
key = 0
# ide de la camara
id_cam = 0
#contador
count = 0

Base = Utils(DATA_PATH, actions, imgSize, size_data)
hands = Base.Hands_model_configuration(False, 1, 1)

k = 0
capture = cv2.VideoCapture(id_cam)

Base.Create_datasets_dir()

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
                Base.Draw_Bound_Boxes(positions, frame)
                resized_hand = Base.Get_image_resized(positions, copie_img)
                cv2.imshow("image", resized_hand)
                if key == 115:
                    # presiona la s para empezar a recolectar
                    k = 1
                if k == 1:
                   Base.Save_resized_hand(resized_hand, count, hand_type)
                   count+=1
        if key == 27:
             exit(0)
        cv2.imshow("image capture", frame)
    capture.release()
    cv2.destroyAllWindows()