# -*- coding:UTF-8 -*-
## 改程序修改于231101，由陈龙修改
## 修改内容包括
## 1、数据窗口长度设置为1
## 2、串口数据第一位被当做
import csv
import time
from statistics import mode

import keyboard
import matplotlib.pyplot as plt
import minimalmodbus
import winsound

# 参数设置(可以调整）
sr = 1000  # 采样率（单位为hz）
window_size = 100  # 时间窗口大小
#T = 3000 # 蜂鸣阈值
T1 = 2275 #800 # 蜂鸣阈值，尖锐
T2 = 4600 #1200 # 蜂鸣阈值，短促，达到尖锐，但不能达到短促
FILENAME = 'CZY_13.csv' #修改文件名
# 例如60kg体重的人，最大支撑的阈值<60kg*100/4=1500, 随意自主支撑（10%）=1500*0.1=150，指定支撑（20%）=1500*0.2=300

# 串口配置（不要调整）
port = 'COM3'  # 串口端口号
baudrate = 38400 #115200  # 波特率
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
# 创建折线图
plt.ion()  # 开启交互模式，实现动态更新
plt.figure()
plt.xlabel('Time')
plt.ylabel('Data')

x_data = []  # 时间轴数据
x1_data = []
y1_data = []  # 第一个元素的数据
y2_data = []  # 第二个元素的数据
y3_data = []  # 第三个元素的数据
y4_data = []  # 第四个元素的数据
y5_data = []  # y5的数据

baseline1 = 0
baseline2 = 0
baseline3 = 0
baseline4 = 0

try:
    # 打开串口连接
    instrument.serial.open()

    while True:
        # 读取 Holding Register 数据
        data = instrument.read_registers(register_address, register_count)

        # 将无符号整数转换为有符号整数
        signed_data = [((x + 2 ** 15) % 2 ** 16 - 2 ** 15) for x in data]

        # 差分数据整形
        output_vector = []

        for i in range(0, len(signed_data) - 1, 2):
            diff = signed_data[i + 1] - signed_data[i]
            output_vector.append(diff)

        # 更新数据列表
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
            y1_data = []  # 第一个元素的数据
            y2_data = []  # 第二个元素的数据
            y3_data = []  # 第三个元素的数据
            y4_data = []  # 第三个元素的数据

        if len(x_data) > window_size:
            y1 = output_vector[2] - baseline1
            y2 = output_vector[5] - baseline2
            y3 = output_vector[8] - baseline3
            y4 = output_vector[11] - baseline4
            y5=y1+y2+y3+y4
            data_save = [y1, y2, y3, y4, y5]  # 保存y5的数据
            print('data_save = ', data_save)
            y1_data.append(y1)
            y2_data.append(y2)
            y3_data.append(y3)
            y4_data.append(y4)
            y5_data.append(y5)

            # -------------------------------------------------------------------
            # 保存数据到 CSV 文件
            with open(FILENAME, 'a', newline='') as csv_file:  #修改文件名 whs_Walking_1.csv 名字_实验任务
               csv_writer = csv.writer(csv_file)
               csv_writer.writerow(data_save)
            # -------------------------------------------------------------------
            # 检查阈值条件:任意一个值大于T1
            if any(y5 > T2 for y5 in data_save):
                # 触发蜂鸣器
                winsound.Beep(500, 200)  # 指定蜂鸣器频率和持续时间，总体阈值
            # 检查阈值条件:任意一个值大于T2
            elif any(y5 > T1 for y5 in data_save):
                # 触发蜂鸣器
                winsound.Beep(1000, 100)  # 指定蜂鸣器频率和持续时间，总体阈值
            # 绘制折线图
            plt.clf()  # 清空当前图像

            # 计算时间窗口起始索引
            x1_data.append(len(x_data) + 1)
            start_index = max(len(x1_data) - window_size, 0)

            # 绘制时间窗口内的数据:：y1=左前腿 ; y2=左后腿；y3=右后腿；y4=右前腿
            plt.plot(x1_data[start_index:], y1_data[start_index:], label='y1_leftfront')
            plt.plot(x1_data[start_index:], y2_data[start_index:], label='y2_leftbehind')
            plt.plot(x1_data[start_index:], y3_data[start_index:], label='y3_rightbehind')
            plt.plot(x1_data[start_index:], y4_data[start_index:], label='y4_rightfront')
            plt.plot(x1_data[start_index:], y5_data[start_index:], label='y5_sum')

            # 添加图例
            plt.legend()
            plt.draw()
            plt.pause(0.01)

            # 检查是否按下'Esc'键退出程序
            if keyboard.is_pressed('Esc'):
                running = False

        # 添加100ms的延迟,数据采集频率从这里调整
        print(len(x_data))
        time.sleep(1 / sr)

except Exception as e:
    print("出现错误:", str(e))

finally:
    # 关闭串口连接
    instrument.serial.close()

# 循环结束后，确保关闭串口连接
instrument.serial.close()


def main():
