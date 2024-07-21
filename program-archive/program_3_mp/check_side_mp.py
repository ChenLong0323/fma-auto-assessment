import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    image = cv2.imread("../../data_test/hand.png")
    print(image.flags.writeable)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image)
    results1 = hands.process(image).multi_hand_landmarks
    results2 = hands.process(image).multi_handedness
    # 提取 landmarks
    landmarks_l = results1[0].landmark
    landmarks_r = results1[1].landmark

    # 使用列表存储坐标
    x_coordinates = [
        landmarks_r[5].x, landmarks_r[17].x,
        landmarks_l[5].x, landmarks_l[17].x
    ]

    y_coordinates = [
        landmarks_r[5].y, landmarks_r[17].y,
        landmarks_l[5].y, landmarks_l[17].y
    ]

    height, width, _ = image.shape
    x_coordinates = [x * width for x in x_coordinates]
    y_coordinates = [y * height for y in y_coordinates]
    # 如果需要单独访问每个坐标值
    r_2_x, r_5_x, l_2_x, l_5_x = x_coordinates
    r_2_y, r_5_y, l_2_y, l_5_y = y_coordinates


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

# cv2.imshow('MediaPipe Hands', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

radius = 20

plt.imshow(image)
plt.scatter(r_2_x, r_2_y, s=radius**2, c='g', marker='o')  # 绿色点
# 这一行的目的是查看x,y的正方向，红色的xy坐标小于绿色
# plt.scatter(r_2_x-100, r_2_y-100, s=radius**2, c='r', marker='o')  # 红色点
plt.scatter(r_5_x, r_5_y, s=radius**2, c='r', marker='o')  # 红色点
plt.title("Image with Points")
plt.axis('off')
plt.show()




