import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    image = cv2.imread("../data_1_test/hand2.jpg")

    print(image.flags.writeable)

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image).multi_hand_landmarks
    results2 = hands.process(image).multi_handedness
    point1 = results[0].landmark[2].x


