import numpy as np
import book_plots as book_plots
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self):
        # 初始状态 (假设为25)
        self.x = np.eye(1) * 25
        
        # 状态转移矩阵
        self.A = np.eye(1)
        
        # 测量矩阵
        self.H = np.eye(1)
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(1) * .05**2
        
        # 测量噪声协方差矩阵
        self.R = np.eye(1) * .13**2
        
        # 误差协方差矩阵
        self.P = np.eye(1) * 1000

        # 卡尔曼增益
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)

    def predict(self):
        # 预测状态
        self.x = self.A @ self.x
        # 预测误差协方差
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x  # 返回预测的状态值

    def update(self, z):
        #更新卡尔曼增益
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        
        # 更新状态
        self.x = self.x + self.K @ (z - self.H @ self.x)
        
        # 更新误差协方差矩阵
        self.P = (np.eye(1) - self.K @ self.H) @ self.P

    def process_measurement(self, measurement):
        measurement = np.array(measurement).reshape(2, 1)
        # 更新和预测步骤
        self.update(measurement)
        return self.predict()  # 返回预测的结果


# # 使用示例
# kf = KalmanFilter()
# N = 50
# actual_value = 16.3
# measurement_std = .13

# estimates = []
# ps = []

# measurements = [(np.random.normal(actual_value, measurement_std)) for i in range(N)]

# for measurement in measurements:
#    #predicted_state = kf.process_measurement(measurement)
#    #print("Predicted state:", predicted_state)
#    estimates.append(kf.process_measurement(measurement))
#    ps.append(kf.P)

# ps = np.array(ps).flatten()
# estimates = np.array(estimates).flatten()

# # plot the filter output and the variance
# book_plots.plot_measurements(measurements)
# book_plots.plot_filter(estimates, var=ps)
# book_plots.show_legend()
# plt.ylim(16, 17)
# book_plots.set_labels(x='step', y='volts')
# plt.show()

# plt.plot(ps)
# plt.title('Variance')
# print(f'Variance converges to {ps[-1]:.3f}')

