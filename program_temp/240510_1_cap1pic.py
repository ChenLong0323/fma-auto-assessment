import cv2

def take_photo(file_path):
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    # 读取一帧图像
    ret, frame = cap.read()
    frame = frame[:, 430:850]

    # 如果成功读取图像
    if ret:
        # 保存图像到指定路径
        cv2.imwrite(file_path, frame)
        print(f"Photo saved to {file_path}")
    else:
        print("Error: Failed to capture photo.")

    # 关闭摄像头
    cap.release()

if __name__ == "__main__":
    file_path = "photo.jpg"  # 保存照片的文件路径
    take_photo(file_path)
