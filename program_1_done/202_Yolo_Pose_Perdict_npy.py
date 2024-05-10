from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载模型
# def main():


if __name__ == "__main__":
    # main()
    model = YOLO("YOLOv8x-pose.pt")
    # path = "C:/000_data/test/rgb"
    # npys = os.listdir(path)
    # for npy in npys:
    #     txt_name = os.path.splitext(npy)[0] + '.txt'
    data = np.load('test_RGB_rgb_1.npy')  # BRGA文件
    bgr_frame = data[:, :, :3]  # 取其中BRG通道
    print(bgr_frame.shape)
    results = model(source=bgr_frame, save=True, save_txt=True, show=True, show_boxes = True)
