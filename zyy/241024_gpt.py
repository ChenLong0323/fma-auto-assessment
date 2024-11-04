# -*- coding:UTF-8 -*-
import csv
import time
from statistics import mode
import threading
import queue

import keyboard
import matplotlib.pyplot as plt
import minimalmodbus
import winsound

# 参数设置(可以调整）
sr = 1000  # 采样率（单位为hz）
window_size = 100  # 时间窗口大小
T1 = 2275  # 蜂鸣阈值，尖锐
T2 = 4600  # 蜂鸣阈值，短促，达到尖锐，但不能达到短促
FILENAME = 'CZY_13.csv'  # 修改文件名

# 串口配置
port = 'COM3'  # 串口端口号
baudrate = 38400  # 波特率
bytesize = 8  # 数据位
parity = 'N'  # 校验位
stopbits = 1  # 停止位

# Modbus 通信参数
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

# 数据队列
data_queue = queue.Queue()

# 图形数据
x_data = []
x1_data = []
y1_data = []
y2_data = []
y3_data = []
y4_data = []
y5_data = []

baseline1 = 0
baseline2 = 0
baseline3 = 0
baseline4 = 0

# 创建折线图
plt.ion()  # 开启交互模式，实现动态更新
plt.figure()
plt.xlabel('Time')
plt.ylabel('Data')


def data_acquisition():
    """串口数据采集线程"""
    global instrument
    try:
        instrument.serial.open()
        while True:
            # 读取 Holding Register 数据
            data = instrument.read_registers(register_address, register_count)

            # 将无符号整数转换为有符号整数
            signed_data = [((x + 2 ** 15) % 2 ** 16 - 2 ** 15) for x in data]

            # 差分数据整形
            output_vector = [signed_data[i + 1] - signed_data[i] for i in range(0, len(signed_data) - 1, 2)]

            # 将数据放入队列中，供处理线程使用
            data_queue.put(output_vector)
            time.sleep(1 / sr)  # 调整采集频率
    except Exception as e:
        print("数据采集错误:", str(e))
    finally:
        instrument.serial.close()


def data_processing():
    """数据处理线程"""
    global baseline1, baseline2, baseline3, baseline4
    while True:
        if not data_queue.empty():
            output_vector = data_queue.get()

            x_data.append(len(x_data) + 1)  # 时间轴自增

            if len(x_data) < window_size:
                y1_data.append(output_vector[2])
                y2_data.append(output_vector[5])
                y3_data.append(output_vector[8])
                y4_data.append(output_vector[11])
                print("读取到的数据:", output_vector)

            if len(x_data) == window_size:
                baseline1 = mode(y1_data)
                baseline2 = mode(y2_data)
                baseline3 = mode(y3_data)
                baseline4 = mode(y4_data)
                y1_data.clear()
                y2_data.clear()
                y3_data.clear()
                y4_data.clear()

            if len(x_data) > window_size:
                y1 = output_vector[2] - baseline1
                y2 = output_vector[5] - baseline2
                y3 = output_vector[8] - baseline3
                y4 = output_vector[11] - baseline4
                y5 = y1 + y2 + y3 + y4
                data_save = [y1, y2, y3, y4, y5]
                print('data_save = ', data_save)

                y1_data.append(y1)
                y2_data.append(y2)
                y3_data.append(y3)
                y4_data.append(y4)
                y5_data.append(y5)

                # 保存数据到 CSV 文件
                with open(FILENAME, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(data_save)

                # 检查阈值条件，触发蜂鸣器
                if any(y5 > T2 for y5 in data_save):
                    winsound.Beep(500, 200)
                elif any(y5 > T1 for y5 in data_save):
                    winsound.Beep(1000, 100)


def plot_update():
    """数据绘制线程"""
    while True:
        if len(x_data) > window_size:
            plt.clf()
            start_index = max(len(x1_data) - window_size, 0)
            plt.plot(x1_data[start_index:], y1_data[start_index:], label='y1_leftfront')
            plt.plot(x1_data[start_index:], y2_data[start_index:], label='y2_leftbehind')
            plt.plot(x1_data[start_index:], y3_data[start_index:], label='y3_rightbehind')
            plt.plot(x1_data[start_index:], y4_data[start_index:], label='y4_rightfront')
            plt.plot(x1_data[start_index:], y5_data[start_index:], label='y5_sum')
            plt.legend()
            plt.draw()
            plt.pause(0.01)


def main():
    # 创建线程
    acquisition_thread = threading.Thread(target=data_acquisition)
    processing_thread = threading.Thread(target=data_processing)
    plotting_thread = threading.Thread(target=plot_update)

    # 启动线程
    acquisition_thread.start()
    processing_thread.start()
    plotting_thread.start()

    # 主线程等待 'Esc' 退出
    try:
        while not keyboard.is_pressed('Esc'):
            time.sleep(0.1)
    except:
        pass
    finally:
        instrument.serial.close()


if __name__ == "__main__":
    main()
