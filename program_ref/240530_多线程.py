import cv2
from ultralytics import YOLO
import time
import threading
import queue

# Load the YOLOv8 model
model = YOLO('../models/yolov8l-pose.pt')

def fps_cal(start_time, frame_count):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 2:
        fps = frame_count / elapsed_time
        start_time = time.time()
        frame_count = 0
    else:
        fps = frame_count / elapsed_time
    return fps, start_time, frame_count

def draw(frame):
    a = 100
    b = 200
    for i in range(20, 201, 20):
        cv2.rectangle(frame, (a + i, a + i), (b + i, b + i), (0, 255, 0), 2)
    return frame

def capture_frames(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

def process_frames(frame_queue, processed_queue, stop_event):
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model(frame, stream=True, conf=0.7, verbose=False)
            for result in results:
                keypoints = result.keypoints
                frame = result.plot()
            processed_queue.put(frame)

def display_frames(processed_queue, start_time, frame_count, stop_event):
    while not stop_event.is_set():
        if not processed_queue.empty():
            frame = processed_queue.get()
            frame = draw(frame)
            fps, start_time, frame_count = fps_cal(start_time, frame_count)
            fps_text = f"FPS: {fps:.2f}"
            print(fps_text)
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("YOLOv8 Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

def main():
    cap = cv2.VideoCapture(0)
    frame_queue = queue.Queue()
    processed_queue = queue.Queue()

    start_time = time.time()
    frame_count = 0

    stop_event = threading.Event()

    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue, stop_event))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, processed_queue, stop_event))
    display_thread = threading.Thread(target=display_frames, args=(processed_queue, start_time, frame_count, stop_event))

    capture_thread.start()
    process_thread.start()
    display_thread.start()

    capture_thread.join()
    process_thread.join()
    display_thread.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
