import csv
import time
from statistics import mode
import threading
import matplotlib.pyplot as plt
import minimalmodbus
import winsound

# 参数设置(可以调整）
sr = 1000  # 采样率（单位为hz）
window_size = 100  # 时间窗口大小
T1 = 2275  # 蜂鸣阈值，尖锐
T2 = 4600  # 蜂鸣阈值，短促，达到尖锐，但不能达到短促
FILENAME = 'CZY_13.csv'  # 修改文件名

# 串口配置（不要调整）
port = 'COM3'  # 串口端口号
baudrate = 38400  # 波特率
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

# 全局变量
x_data = []
y1_data = []
y2_data = []
y3_data = []
y4_data = []
y5_data = []
baseline1 = 0
baseline2 = 0
baseline3 = 0
baseline4 = 0

# 读取数据的线程函数
def read_data():
    global x_data, y1_data, y2_data, y3_data, y4_data, baseline1, baseline2, baseline3, baseline4
    instrument.serial.open()
    while True:
        data = instrument.read_registers(register_address, register_count)
        signed_data = [((x + 2 ** 15) % 2 ** 16 - 2 ** 15) for x in data]
        output_vector = []
        for i in range(0, len(signed_data) - 1, 2):
            diff = signed_data[i + 1] - signed_data[i]
            output_vector.append(diff)
        x_data.append(len(x_data) + 1)
        if len(x_data) < window_size:
            y1_data.append(output_vector[2])
            y2_data.append(output_vector[5])
            y3_data.append(output_vector[8])
            y4_data.append(output_vector[11])
        elif len(x_data) == window_size:
            baseline1 = mode(y1_data)
            baseline2 = mode(y2_data)
            baseline3 = mode(y3_data)
            baseline4 = mode(y4_data)
            y1_data = []
            y2_data = []
            y3_data = []
            y4_data = []
        time.sleep(1 / sr)

# 处理数据和绘图的线程函数
def process_data():
    global x_data, y1_data, y2_data, y3_data, y4_data, y5_data
    plt.ion()
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Data')
    try:
        while True:
            if len(x_data) > window_size:
                y1 = y1_data[-1] - baseline1
                y2 = y2_data[-1] - baseline2
                y3 = y3_data[-1] - baseline3
                y4 = y4_data[-1] - baseline4
                y5 = y1 + y2 + y3 + y4
                y5_data.append(y5)
                data_save = [y1, y2, y3, y4, y5]
                with open(FILENAME, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(data_save)
                if any(y > T2 for y in data_save):
                    winsound.Beep(500, 200)
                elif any(y > T1 for y in data_save):
                    winsound.Beep(1000, 100)
                plt.clf()
                x1_data = [len(x_data) + 1]
                start_index = max(len(x1_data) - window_size, 0)
                plt.plot(x1_data[start_index:], y1_data[start_index:], label='y1_leftfront')
                plt.plot(x1_data[start_index:], y2_data[start_index:], label='y2_leftbehind')
                plt.plot(x1_data[start_index:], y3_data[start_index:], label='y3_rightbehind')
                plt.plot(x1_data[start_index:], y4_data[start_index:], label='y4_rightfront')
                plt.plot(x1_data[start_index:], y5_data[start_index:], label='y5_sum')
                plt.legend()
                plt.draw()
                plt.pause(0.01)
            time.sleep(0.01)
    except KeyboardInterrupt:
        instrument.serial.close()

# 创建并启动线程
read_thread = threading.Thread(target=read_data)
process_thread = threading.Thread(target=process_data)
read_thread.start()
process_thread.start()

# 等待线程结束
read_thread.join()
process_thread.join()