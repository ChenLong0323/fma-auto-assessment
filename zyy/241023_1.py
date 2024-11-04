"""
问题：去除ext过程，直接从keypoints获取点作为交互点
"""
import threading
from collections import deque
import queue
import csv
import time
from statistics import mode

import keyboard
import matplotlib.pyplot as plt
import minimalmodbus
import winsound
import logging


def initialize():
    # 参数设置(可以调整）
    sr = 1000  # 采样率（单位为hz）
    window_size = 100  # 时间窗口大小
    # T = 3000 # 蜂鸣阈值
    T1 = 2275  # 800 # 蜂鸣阈值，尖锐
    T2 = 4600  # 1200 # 蜂鸣阈值，短促，达到尖锐，但不能达到短促
    FILENAME = 'CZY_13.csv'  # 修改文件名
    # 例如60kg体重的人，最大支撑的阈值<60kg*100/4=1500, 随意自主支撑（10%）=1500*0.1=150，指定支撑（20%）=1500*0.2=300

    # 串口配置（不要调整）
    port = 'COM3'  # 串口端口号
    baudrate = 38400  # 115200  # 波特率
    bytesize = 8  # 数据位
    parity = 'N'  # 校验位：N 表示无校验
    stopbits = 1  # 停止位

    # Modbus 通信参数（不要调整）
    slave_address = 1  # 从设备地址
    register_address = 500  # 起始地址
    register_count = 24  # 读取长度

    # 创建 Modbus 通信对象
    instrument = minimalmodbus.Instrument(port, slave_address)
    instrument.serial.baudrate = baudrate
    instrument.serial.bytesize = bytesize
    instrument.serial.parity = parity
    instrument.serial.stopbits = stopbits
    instrument.serial.timeout = 0.1  # 通信超时时间
    instrument.serial.close()
    return instrument, register_address, register_count


def get_raw_data(instrument, register_address, register_count,
                 flag_running, deque_fps_raw_data,
                 queue_data_raw):
    while flag_running.is_set():
        try:
            # 计算FPS
            deque_fps_raw_data.append(time.time())
            if len(deque_fps_raw_data) > 30:
                deque_fps_raw_data.popleft()
            if len(deque_fps_raw_data) > 1:
                fps = len(deque_fps_raw_data) / (deque_fps_raw_data[-1] - deque_fps_raw_data[0])
                logging.info(f"FPS_cap: {fps:.2f}")

            # todo 函数内容
            data = instrument.read_registers(register_address, register_count)
            try:
                queue_data_raw.put_nowait(data)
            except queue.Full:
                continue  # 直接继续，不记录日志以减少影响


        except Exception as e:
            logging.error(f"Error in func_frame_cap: {str(e)}")
            break


def process_data(flag_running, deque_fps_processed_data,
                 queue_data_raw, queue_data_processed):
    while flag_running.is_set():
        try:
            # 计算FPS
            deque_fps_processed_data.append(time.time())
            if len(deque_fps_processed_data) > 30:
                deque_fps_processed_data.popleft()
            if len(deque_fps_processed_data) > 1:
                fps = len(deque_fps_processed_data) / (deque_fps_processed_data[-1] - deque_fps_processed_data[0])
                logging.info(f"FPS_cap: {fps:.2f}")

            # todo 函数内容
            data = queue_data_raw.get()
            # 将无符号整数转换为有符号整数
            signed_data = [((x + 2 ** 15) % 2 ** 16 - 2 ** 15) for x in data]

            # 差分数据整形
            output_vector = None

            for i in range(0, len(signed_data) - 1, 2):
                diff = signed_data[i + 1] - signed_data[i]
                output_vector = diff

            if output_vector is not None:
                try:
                    queue_data_processed.put_nowait(output_vector)
                except queue.Full:
                    continue  # 直接继续，不记录日志以减少影响

        except Exception as e:
            logging.error(f"Error in func_frame_cap: {str(e)}")
            break


def main():
    instrument, register_address, register_count = initialize()

    instrument.serial.open()

    queue_data_raw = queue.Queue(maxsize=3)
    queue_data_processed = queue.Queue(maxsize=3)

    deque_fps_raw_data = deque(maxlen=30)
    deque_fps_processed_data = deque(maxlen=30)
    deque_fps_draw = deque(maxlen=30)

    # 程序是否运行
    flag_running = threading.Event()
    flag_running.set()  # Initially set the event to True

    thread_get_raw_data = threading.Thread(target=get_raw_data, args=(queue_frame_image, queue_keypoints, deque_fps_raw_data,
                                                                      data_cons, flag_running, action_idx))
    thread_process_data = threading.Thread(target=process_data, args=(queue_frame_image, queue_keypoints, deque_fps_raw_data,
                                                                      data_cons, flag_running, action_idx))

    thread_get_raw_data.start()

    try:
        while flag_running.is_set():
            print('main is running')
            time.sleep(0.01)
    except KeyboardInterrupt:
        flag_running.clear()  # Set the event to False to signal all threads to stop

    thread_get_raw_data.join()


if __name__ == "__main__":
    main()
