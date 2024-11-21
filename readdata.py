import serial
import glob
from kalman import KalmanFilter
import re
import numpy as np

def extract_numbers(data):
    """从字符串中提取数字"""
    # 使用正则表达式提取所有浮点数
    numbers = re.findall(r'-?\d+\.?\d*', data)
    # 转换为浮点数
    numbers = [float(num) for num in numbers]
    
    # 将数字对应到各个传感器值
    gyro_roll, acc_roll, gyro_pitch, acc_pitch, gyro_yaw = numbers
    return gyro_roll, acc_roll, gyro_pitch, acc_pitch, gyro_yaw


def find_usb_port():
    """查找可能的USB串口设备"""
    # Mac上常见的USB串口路径
    ports = glob.glob('/dev/tty.usb*') + glob.glob('/dev/tty.wch*') + glob.glob('/dev/cu.usb*')
    return ports[0] if ports else None


kf_roll = KalmanFilter()
kf_pitch = KalmanFilter()

kf_roll.P = np.eye(1) * 0.05**2
kf_roll.x = np.zeros((1, 1))
kf_roll.A = np.eye(1)
kf_roll.H = np.array([[1],
                [1]])
kf_roll.Q = np.eye(1) * .02**2
kf_roll.R = np.array([[8**2, 0],
                 [0, .02**2]])

kf_pitch.P = np.eye(1) * 0.05**2
kf_pitch.x = np.zeros((1, 1))
kf_pitch.A = np.eye(1)
kf_pitch.H = np.array([[1],
                [1]])
kf_pitch.Q = np.eye(1) * .02**2
kf_pitch.R = np.array([[8**2, 0],
                 [0, .02**2]])



try:
    # 自动查找USB端口
    port = find_usb_port()
    if not port:
        print("未找到USB设备")
        exit()
        
    print(f"找到端口: {port}")
    
    # 打开串口，波特率115200
    ser = serial.Serial(port, 115200)
    print("连接成功，开始读取数据...")
    
    # 持续读取数据
    while True:
        if ser.in_waiting:  # 如果有数据等待读取
            line = ser.readline().decode().strip()  # 读取一行数据
            print(line)
            #gyro_roll, acc_roll, gyro_pitch, acc_pitch, gyro_yaw = extract_numbers(line)
            #roll_fusion = kf_roll.process_measurement([gyro_roll, acc_roll])
            #pitch_fusion = kf_pitch.process_measurement([gyro_pitch, acc_pitch])

            #roll_fusion = roll_fusion.flatten()[0]
            #pitch_fusion = pitch_fusion.flatten()[0]

            #K_roll = kf_roll.K
            #K_pitch = kf_pitch.K
            
            #print(f"roll:{roll_fusion:.1f} pitch:{pitch_fusion:.1f} roll gain:{K_roll} pitch gain:{K_pitch}")

            
            
except KeyboardInterrupt:
    print("\n程序结束")
    ser.close()  # 关闭串口
except Exception as e:
    print(f"错误: {e}")