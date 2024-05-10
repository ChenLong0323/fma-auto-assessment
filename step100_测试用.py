import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
cap = cv2.VideoCapture(0)
# FPS 计算
frame_count = 0
start_time = time.time()
fps = 0

while cap.isOpened():
    success, image = cap.read()
    image = image[:, 430:850]  # 1280 - 420 = 860, 860 / 2 = 430
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.q
        continue
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            # 手部标志模型的复杂度：0或1。随着模型复杂度的增加，标志的准确性以及推理延迟通常会增加。默认为1
            model_complexity=0,
            min_detection_confidence=0.5,
            # 手部标志跟踪模型的最小置信度值（[0.0, 1.0]），以使手部标志被成功跟踪，否则将自动在下一个输入图像上调用手部检测。
            # 将其设置为较高的值可以增加解决方案的鲁棒性，但会增加延迟.
            min_tracking_confidence=0.5) as hands:  #
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
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
    # Flip the image horizontally for a selfie-view display.
    annotated_image = cv2.flip(image, 1)
    cv2.putText(annotated_image, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_count += 1
    # Calculate and print FPS every second
    if frame_count % 30 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 30 / elapsed_time
        print(f"FPS: {fps}")
        # Reset variables for the next second
        frame_count = 0
        start_time = time.time()
cap.release()
