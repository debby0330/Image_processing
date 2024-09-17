# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:21:09 2024

@author: user
"""

from numpy import zeros, clip, int16, uint8, ravel, where,zeros_like
from pandas import DataFrame, ExcelWriter
from openpyxl.chart import BarChart, Reference
from cv2 import imshow, resize, INTER_NEAREST, waitKey, destroyAllWindows
from glob import glob
from os.path import join, basename

# 將字符轉換為灰階值
def char_to_gray(char):
    if '0' <= char <= '9':
        return ord(char) - ord('0')  # '0'-'9' 轉換為 0-9
    elif 'A' <= char <= 'V':
        return ord(char) - ord('A') + 10  # 'A'-'V' 轉換為 10-31
    else:
        return None  # 若非上述字符，忽略

# 從 .64 文件中讀取數據並轉換為 64x64 的二維矩陣
def read_64_image(file_path):
    with open(file_path, 'r', encoding='ascii') as f:
        data = f.read()

    data = ''.join(char for char in data if char in '0123456789ABCDEFGHIJKLMNOPQRSTUV')
    expected_length = 64 * 64
    if len(data) != expected_length:
        raise ValueError(f"文件 {file_path} 不符合 64x64 圖像。")

    image = zeros((64, 64), dtype=int16)
    index = 0
    for row in range(64):
        for col in range(64):
            if index < len(data):
                gray_value = char_to_gray(data[index])
                if gray_value is not None:
                    image[row, col] = gray_value
                index += 1

    return clip(image, 0, 31)

# 顯示圖像
def display_image(image, title='Image'):
    image_255 = image * 8
    image_255 = image_255.astype(uint8)
    scaled_image = resize(image_255, (512, 512), interpolation=INTER_NEAREST)
    imshow(title, scaled_image)
    waitKey(0)
    destroyAllWindows()

# 計算圖像的灰階直方圖
def calculate_histogram(image):
    hist = zeros(32, dtype=int)
    for gray_value in ravel(image):
        hist[gray_value] += 1
    return hist

# 保存直方圖數據到 Excel 文件並插入圖表
def save_histogram_to_excel(histograms, output_file):
    with ExcelWriter(output_file, engine='openpyxl') as writer:
        workbook = writer.book
        for i, hist in enumerate(histograms):
            gray_levels = list(range(32))
            histogram_df = DataFrame({
                'Gray Level': gray_levels,
                'Pixel Count': hist
            })
            sheet_name = f'Histogram_{i+1}'
            histogram_df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            chart = BarChart()
            chart.title = f"Gray Level Histogram {i+1}"
            chart.style = 2
            chart.x_axis.title = 'Gray Level'
            chart.y_axis.title = 'Pixel Count'
            data = Reference(worksheet, min_col=2, min_row=1, max_col=2, max_row=33)
            categories = Reference(worksheet, min_col=1, min_row=2, max_row=33)
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(categories)
            worksheet.add_chart(chart, "E5")

# 影像處理操作：加法、減法或乘法
def cal_constant(image, constant, operation='add'):
    if operation == 'add':
        # 加法操作，並限制像素值範圍
        return clip(image + constant, 0, 31)
    elif operation == 'subtract':
        # 減法操作，確保不低於 0
        result = image - constant
        result = where(result < 0, 0, result)
        return clip(result, 0, 31)
    elif operation == 'multiply':
        # 乘法操作，並限制像素值範圍
        return clip(image * constant, 0, 31)
    else:
        raise ValueError("無效的操作類型，請使用 'add'、'subtract' 或 'multiply'")

# 邊緣檢測操作：計算 g(x, y) = f(x, y) - f(x-1, y)
def edge_detection(image):
    height, width = image.shape
    edge_image = zeros_like(image, dtype=int16)
    
    # 計算邊緣圖像
    edge_image[:, 1:] = image[:, 1:] - image[:, :-1]
    
    # 將負值設為0
    edge_image = clip(edge_image, 0, 31)
    
    return edge_image

# 設定 .64 文件的路徑和輸出文件
def read_config(file_path):
    with open(file_path) as file:
        return dict(line.strip().split('=', 1) for line in file)

# 讀取配置文件
config = read_config(r".\\config.txt")

# 使用配置文件中的路徑
input_folder = config['input_folder']
output_file = config['output_file']

# 讀取所有 .64 文件的路徑
file_paths = glob(join(input_folder, '*.64'))

# 記錄 JET.64 和 LIBERTY.64 的影像
image1 = None
image2 = None

histograms = []

if not file_paths:
    print(f"在資料夾 {input_folder} 中未找到任何 .64 文件")
else:
    for file_path in file_paths:
        try:
            # 讀取圖像
            image = read_64_image(file_path)
            
            # 顯示原始圖像
            display_image(image, "Original Image")
            
            # 計算原始圖像的直方圖
            histograms.append(calculate_histogram(image))
            
            # 對圖像加常數 15
            image_add_15 = cal_constant(image, 15, 'add')
            display_image(image_add_15, "Image + 15")
            histograms.append(calculate_histogram(image_add_15))
            
            # 對圖像減常數 7
            image_subtract_7 = cal_constant(image, 7, 'subtract')
            display_image(image_subtract_7, "Image - 7")
            histograms.append(calculate_histogram(image_subtract_7))
            
            # 對圖像乘常數 10
            image_multiply_10 = cal_constant(image, 10, 'multiply')
            display_image(image_multiply_10, "Image * 10")
            histograms.append(calculate_histogram(image_multiply_10))

            # 記錄 JET.64 和 LIBERTY.64 影像
            if 'JET.64' in basename(file_path):
                image1 = image
            elif 'LIBERTY.64' in basename(file_path):
                image2 = image

            print(f"已處理文件 {file_path}")

        except Exception as e:
            print(f"處理文件 {file_path} 時發生錯誤: {e}")

    if histograms:
        # 保存直方圖到 Excel 並插入圖表
        save_histogram_to_excel(histograms, output_file)
        print(f"直方圖已保存到 {output_file}")

    # 確保有 image1 和 image2 
    if image1 is not None and image2 is not None:
        # 計算圖像平均
        average_image = (image1 + image2) / 2
        average_image = clip(average_image, 0, 31).astype(int16)
        
        # 顯示平均圖像
        display_image(average_image, "Average Image of JET and LIBERTY")
        
        # 計算平均圖像的直方圖
        average_histogram = calculate_histogram(average_image)
        histograms.append(average_histogram)

        # 將平均圖像直方圖存到 Excel 
        save_histogram_to_excel(histograms, output_file)
        print(f"直方圖已更新並保存到 {output_file}")

    # 對4張影像進行邊緣偵測並顯示
    for file_path in file_paths:
        try:
            image = read_64_image(file_path)
            
            # 計算邊緣檢測圖像
            edge_image = edge_detection(image)
            display_image(edge_image, f"Edge Detection of {basename(file_path)}")
            
            # 計算邊緣檢測圖像的直方圖
            edge_histogram = calculate_histogram(edge_image)
            histograms.append(edge_histogram)

            print(f"邊緣檢測已完成文件 {file_path}")

        except Exception as e:
            print(f"處理邊緣檢測文件 {file_path} 時發生錯誤: {e}")

    if histograms:
        # 輸出邊緣檢測圖像的直方圖到 Excel 
        save_histogram_to_excel(histograms, output_file)
        print(f"直方圖已更新並保存到 {output_file}")












