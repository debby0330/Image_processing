# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:02:20 2024

@author: user
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QSlider, QVBoxLayout, QHBoxLayout, QPushButton, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QFont
from PyQt5.QtCore import Qt
import numpy as np
import os

# 讀取.64檔案，並將字符轉換為灰階值，顯示影像
def read_64_image(file_path):
    try:
        with open(file_path, 'r', encoding='ascii') as f:
            data = f.read()

        data = ''.join(char for char in data if char in '0123456789ABCDEFGHIJKLMNOPQRSTUV')
        expected_length = 64 * 64
        if len(data) != expected_length:
            raise ValueError(f"File {file_path} does not符合 64x64 影像尺寸。")

        image = np.zeros((64, 64), dtype=np.int16)
        index = 0
        for row in range(64):
            for col in range(64):
                if index < len(data):
                    gray_value = char_to_gray(data[index])
                    if gray_value is not None:
                        image[row, col] = gray_value
                    index += 1

        return np.clip(image, 0, 31)
    except Exception as e:
        raise RuntimeError(f"讀取影像檔案 {file_path} 發生錯誤: {e}")

# 字符轉換為灰階值0-9,A-V to 32灰階
def char_to_gray(char):
    if '0' <= char <= '9':
        return ord(char) - ord('0')
    elif 'A' <= char <= 'V':
        return ord(char) - ord('A') + 10
    else:
        return None

# 計算影像灰階值及pixel數，繪製直方圖
def calculate_histogram(image):
    hist = np.zeros(32, dtype=int)
    for gray_value in image.ravel():
        hist[gray_value] += 1
    return hist

# 顯示影像，將32灰階轉換為256灰階
def display_image(image):
    try:
        image_255 = image * 8
        image_255 = image_255.astype(np.uint8)
        height, width = image_255.shape
        qimage = QImage(image_255.data, width, height, width, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage)
    except Exception as e:
        raise RuntimeError(f"從影像資料建立 QPixmap 發生錯誤: {e}")

# 後一行減前一行灰階值，顯示影像輪廓
def edge_detection(image):
    height, width = image.shape
    edge_image = np.zeros_like(image, dtype=np.int16)
    edge_image[:, 1:] = image[:, 1:] - image[:, :-1]
    edge_image = np.clip(edge_image, 0, 31)
    return edge_image

# 兩張影像的灰階值平均
def average_images(image1, image2):
    return np.clip((image1 + image2) // 2, 0, 31)

# QT介面設計
class ImageDisplayApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.combo_box = QComboBox()
        self.combo_box.addItems(['Please select one image', 'JET.64', 'LIBERTY.64', 'LINCOLN.64', 'LISA.64'])
        self.combo_box.currentIndexChanged.connect(self.load_image)

        self.tool_combo_box = QComboBox()
        self.tool_combo_box.addItems(['Select tools', '+', '-', '*', 'Show Edge'])
        self.tool_combo_box.currentIndexChanged.connect(self.apply_tool)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(31)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)

        self.image_label = QLabel()
        self.histogram_label = QLabel()

        self.apply_button = QPushButton('Apply')
        self.apply_button.clicked.connect(self.apply_tool)

        self.average_button = QPushButton('Average Image')  # 平均影像按鈕
        self.average_button.clicked.connect(self.average_image)  # 接到平均影像功能

        # 布局設定
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.combo_box)
        layout.addLayout(top_layout)

        layout.addWidget(self.average_button)

        tool_layout = QHBoxLayout()
        tool_layout.addWidget(self.tool_combo_box)
        tool_layout.addWidget(self.slider)
        tool_layout.addWidget(self.apply_button)
        layout.addLayout(tool_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.image_label)
        h_layout.addWidget(self.histogram_label)
        layout.addLayout(h_layout)

        self.setLayout(layout)

        self.setWindowTitle('影像與直方圖顯示')
        self.setGeometry(300, 300, 1200, 600)
        self.show()

        self.current_image = None

    def read_config(self):
        """從 config.txt 讀取 input_folder 路徑"""
        config_file = 'config.txt'
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, 'r') as file:
            for line in file:
                if line.startswith('input_folder='):
                    return line.strip().split('=')[1]
        raise ValueError('input_folder not found in config.txt')

    # 讀取資料夾中影像
    def load_image(self):
        image_name = self.combo_box.currentText()
        if image_name == 'Please select one image':
            self.image_label.clear()
            self.histogram_label.clear()
        else:
            # 從 config.txt 讀取資料夾路徑
            input_folder = self.read_config()
            # 合併 input_folder 和 image_name 成完整的檔案路徑
            file_path = os.path.join(input_folder, image_name)
            if not os.path.isfile(file_path):
                self.image_label.setText(f"檔案未找到: {file_path}")
                self.histogram_label.clear()
                return
            
            try:
                self.current_image = read_64_image(file_path)
                pixmap = display_image(self.current_image)
                self.image_label.setPixmap(pixmap)

                # 繪製直方圖
                hist = calculate_histogram(self.current_image)
                histogram_pixmap = self.draw_histogram(hist)
                self.histogram_label.setPixmap(histogram_pixmap)
            except Exception as e:
                self.image_label.setText(f"載入影像錯誤: {e}")
                self.histogram_label.clear()

    # 設計QT影像處理的功能選單，運算常數範圍為0-31
    def apply_tool(self):
        if self.current_image is None:
            return

        tool = self.tool_combo_box.currentText()
        constant = self.slider.value()

        if tool == '+':
            result_image = np.clip(self.current_image + constant, 0, 31)
        elif tool == '-':
            result_image = np.clip(self.current_image - constant, 0, 31)
        elif tool == '*':
            result_image = np.clip(self.current_image * constant, 0, 31)
        elif tool == 'Show Edge':
            result_image = edge_detection(self.current_image)
        else:
            result_image = self.current_image

        # 輸出影像和顯示直方圖
        pixmap = display_image(result_image)
        self.image_label.setPixmap(pixmap)

        hist = calculate_histogram(result_image)
        histogram_pixmap = self.draw_histogram(hist)
        self.histogram_label.setPixmap(histogram_pixmap)
        
    # average影像功能，可選擇兩張影像
    def average_image(self):
        images = ['JET.64', 'LIBERTY.64', 'LINCOLN.64', 'LISA.64']
        image1_name, ok1 = QInputDialog.getItem(self, "選擇第一張影像", "影像", images, 0, False)
        image2_name, ok2 = QInputDialog.getItem(self, "選擇第二張影像", "影像", images, 0, False)

        if ok1 and ok2:
            try:
                input_folder = self.read_config()
                image1 = read_64_image(os.path.join(input_folder, image1_name))
                image2 = read_64_image(os.path.join(input_folder, image2_name))

                result_image = average_images(image1, image2)
                pixmap = display_image(result_image)
                self.image_label.setPixmap(pixmap)
            
                # 計算average image的直方圖
                hist = calculate_histogram(result_image)
                histogram_pixmap = self.draw_histogram(hist)
                self.histogram_label.setPixmap(histogram_pixmap)
            except Exception as e:
                self.image_label.setText(f"計算平均影像時出錯: {e}")
                self.histogram_label.clear()

    # 直方圖顯示樣式
    def draw_histogram(self, hist):
        width = 256
        height = 200
        margin = 30
        image = QImage(width, height, QImage.Format_RGB888)
        image.fill(Qt.white)
        painter = QPainter(image)
        pen = QPen(Qt.black)
        painter.setPen(pen)
        brush = QBrush(Qt.black)
        painter.setBrush(brush)

        bar_width = (width - 2 * margin) // len(hist)
        max_value = max(hist)
        scale = (height - 2 * margin) / max_value

        # 繪製直方圖
        for i, value in enumerate(hist):
            bar_height = int(value * scale)
            x = margin + i * bar_width
            y = height - margin - bar_height
            painter.drawRect(x, y, bar_width - 1, bar_height)

        # 繪製 x 軸標題
        painter.setFont(QFont('Arial', 12))
        painter.drawText(width // 2 - 50, height - 5, "Gray Level 0-31")

        # 繪製 y 軸標題
        painter.save()
        painter.translate(margin // 2, height // 2)
        painter.rotate(-90)
        painter.drawText(0, 0, "Pixel Count")
        painter.restore()

        painter.end()
        return QPixmap.fromImage(image)

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 關閉括號
    window = ImageDisplayApp()
    sys.exit(app.exec_())

