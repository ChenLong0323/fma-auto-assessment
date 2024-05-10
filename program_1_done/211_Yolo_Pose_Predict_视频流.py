import cv2
from ultralytics import YOLO
import time

start_time = time.time()
# Load the YOLOv8 model
model = YOLO('models/yolov8l-pose.pt')

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # FPS 计算
    frame_count = 0
    start_time = time.time()
    fps = 0


    while True:
        ret, frame = cap.read()
        # 读取摄像头画面
        if ret:
            # Run YOLOv8 inference on the frame
            results = model(frame, stream=True, conf=0.7)
            for result in results:
                keypoints = result.keypoints
                frame = result.plot()
            # Visualize the results on the frame

            # Display the annotated frame
            fps_text = f"FPS: {fps}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("YOLOv8 Inference", frame)

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
                # Reset variables for the next second
                frame_count = 0
                start_time = time.time()


    # 释放摄像头资源
    cap.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()