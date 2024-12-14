import cv2
import numpy as np
import matplotlib.pyplot as plt
from kalman import KalmanFilter
import book_plots as book_plots

kf = KalmanFilter()
kf.x = np.array([[-0.8],
                [250]])
kf.A = np.eye(2)
kf.H = np.array([[0.5, 0],
                 [0, 30]])
kf.Q = np.array([[0.05, 0],
                 [0, 5]])
kf.R = np.array([[0.02, 0],
                 [0, 5]])
kf.P = np.eye(2) * 1

def changecolor(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([190, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    image[red_mask > 0 ] = [186, 186, 186]

    return image

def canny(image, gaussian_range, lower_th, higher_th, L2gradient):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, gaussian_range, 0)
    canny = cv2.Canny(blur, lower_th, higher_th, L2gradient=L2gradient)

    return canny    

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    rec = np.array([[(500, height//2), (500, height), (3500, height), (3500, height//2)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rec, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        try:
            for line in lines:
                # 确保坐标是整数
                x1, y1, x2, y2 = map(int, line.reshape(4))
                
                # 检查坐标是否在有效范围内
                height, width = image.shape[:2]
                x1 = max(0, min(x1, width-1))
                x2 = max(0, min(x2, width-1))
                y1 = max(0, min(y1, height-1))
                y2 = max(0, min(y2, height-1))
                
                # 绘制线段
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                
        except Exception as e:
            print(f"绘制线段时出错: {str(e)}")
            print(f"线段数据: {lines}")
            
    return line_image

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    except Exception:
        return None

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None
    
    # 添加边界值
    min_slope = 0.8       # 最小斜率阈值
    max_slope = 15     # 最大斜率阈值
    image_center = image.shape[1] / 2
    
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            
            # 避免垂直线
            if x2 - x1 == 0:
                continue
                
            # 计算斜率和截距
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            # 使用更严格的斜率过滤
            if not (min_slope < abs(slope) < max_slope):
                continue
                
            # 计算线段中点
            mid_x = (x1 + x2) / 2
            
            # 计算x截距（与底部的交点）
            x_intercept = (image.shape[0] - intercept) / slope
            
            # 基于位置和斜率方向的分类
            if  mid_x < image_center-250:  # 左车道线
                left_fit.append((slope, intercept))
            elif mid_x > image_center+250:  # 右车道线
                right_fit.append((slope, intercept))
        
        # 确保有足够的线段
        if len(left_fit) < 2 or len(right_fit) < 2:
            return None
            
        # 过滤异常值(停用过滤)
        #filtered_left_fit = filter_outliers(left_fit)
        #filtered_right_fit = filter_outliers(right_fit)
        
        filtered_left_fit = left_fit
        filtered_right_fit = right_fit

        # 再次检查过滤后的数量
        if not filtered_left_fit or not filtered_right_fit:
            return None
            
        # 计算平均值
        left_fit_average = np.average(filtered_left_fit, axis=0)
        right_fit_average = np.average(filtered_right_fit, axis=0)

        # 卡尔曼滤波
        
        left_fit_average = kf.process_measurement(left_fit_average)
        
        # 生成最终坐标
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        
        if left_line is None or right_line is None:
            return None
            
        return np.array([left_line, right_line])
        
    except Exception as e:
        print(f"处理车道线时出错: {str(e)}")
        return None

def filter_outliers(fit_points):
    # 边界检查
    if not fit_points or len(fit_points) < 2:
        return fit_points
        
    try:
        # 转换为numpy数组
        fit_array = np.array(fit_points)
        
        # 检查数组维度
        if fit_array.ndim == 1:
            fit_array = fit_array.reshape(-1, 2)
            
        slopes = fit_array[:, 0]
        intercepts = fit_array[:, 1]
        
        # 计算斜率的绝对值
        abs_slopes = np.abs(slopes)
        
        # 斜率筛选 - 保留接近目标斜率的线段
        target_slope = 100  # 预期的斜率绝对值
        slope_variances = np.abs(abs_slopes - target_slope)
        slope_threshold = np.percentile(slope_variances,75)
        slope_mask = slope_variances <= slope_threshold
        
        # 计算x截距（与x轴的交点）
        x_intercepts = -1 * intercepts / slopes  # y = mx + b, 当y=0时，x = -b/m
        
        # x截距筛选 - 保留接近图像中心的线段
        target_x = 2000  # 预期的x截距
        x_variances = np.abs(x_intercepts - target_x)
        x_threshold = np.percentile(x_variances, 75)
        x_mask = x_variances <= x_threshold
        
        final_mask = slope_mask & x_mask
        filtered_points = fit_array[final_mask]
        
        # 确保至少保留一些点
        if len(filtered_points) < 2:
            # 如果过滤太严格，只按斜率筛选
            filtered_points = fit_array[slope_mask]
            
        return filtered_points.tolist()
        
    except Exception as e:
        print(f"过滤异常值时出错: {str(e)}")
        return fit_points  # 出错时返回原始数据


image = cv2.imread('/Users/wumingyao/Documents/lab7/autopilot/data/videotest.jpg')
lane_image = np.copy(image)



##changecolor = changecolor(lane_image)

cap = cv2.VideoCapture('/Users/wumingyao/Documents/lab7/autopilot/data/videotest.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame, (7,7), 80, 150, True)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 30, minLineLength=10, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.3, line_image, 1, 1, 1)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 800, 600)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
