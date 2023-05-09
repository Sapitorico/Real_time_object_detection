#!/usr/bin/python
from Dinamic_model import LSTM_Model
from Base_Model.Base_model_detection import Base_Model
import cv2
import  numpy as np

cap = cv2.VideoCapture(2)
actions = np.array(['hola', 'como estas', 'gracias', 'muchas gracias'])
sequence = []
sentence = []
predictions = []
threshold = 0.5
colors = []
for i in range(len(actions)):
    colors.append((245,117,16))
print(len(colors))

detect = Base_Model()
LSTM = LSTM_Model()
model = LSTM.model


with detect.holistic_model as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        # Make detections
        image, results = detect.mediapipe_detection(frame, holistic)
        # Draw landmarks
        # detect.draw_landmarks(image, results)

        image = cv2.flip(image, 1)
        # Check if at least one hand is present in the results
        if results.left_hand_landmarks or results.right_hand_landmarks:
            # 2. Prediction logic
            keypoints = detect.extract_keypoints(results)
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

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 10:
                    sentence = sentence[-10:]

                # Viz probabilities
                # frame = LSTM.prob_viz(res, actions, frame, colors)
                print(f"Prediction probability: {res[np.argmax(res)] * 100:.2f}%")
        # Dibujar rectángulo con opacidad del 50%
        rect_color = (245, 117, 16)
        rect_thickness = -1
        rect_opacity = 0.5
        rect_image = np.zeros_like(image)
        cv2.rectangle(rect_image, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), rect_color,
                      rect_thickness)
        image = cv2.addWeighted(image, 1 - rect_opacity, rect_image, rect_opacity, 0)
        cv2.putText(image, ' '.join(sentence), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()