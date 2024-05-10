import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
cap = cv2.VideoCapture(1)

# FPS 计算
frame_count = 0
start_time = time.time()
fps = 0

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.q
            continue

        image = image[:, 430:850]  # 1280 - 420 = 860, 860 / 2 = 430
        print(image.flags.writeable)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # 图片反转
        annotated_image = cv2.flip(image, 1)

        # 绘制图片及FPS
        cv2.putText(annotated_image, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Hands', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1

        els_time = time.time() - start_time
        if els_time > 2:
            # 计算FPS
            fps = frame_count / els_time
            start_time = time.time()
            frame_count = 0
    cap.release()
