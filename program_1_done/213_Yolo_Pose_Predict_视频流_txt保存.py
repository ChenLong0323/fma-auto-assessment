import cv2
from ultralytics import YOLO
import time

start_time = time.time()
# Load the YOLOv8 model
model = YOLO('models/yolov8n-pose.pt')

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # FPS 计算
    frame_count = 0
    start_time = time.time()


    while True:
        ret, frame = cap.read()
        # 读取摄像头画面
        if ret:
            # Run YOLOv8 inference on the frame
            results = model(frame, stream=True)
            for result in results:
                keypoints = result.keypoints  # Keypoints object for pose outputs
                annotated_frame = result.plot()

                # 数据保存
                filename = str(frame_count) + '.txt'
                result.save_txt(txt_file=filename, save_conf=False)

            # 图片绘制
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            #  计算FPS过程
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = 30 / elapsed_time
                print(f"FPS: {fps}")
                start_time = end_time


    # 释放摄像头资源
    cap.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()