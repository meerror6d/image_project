# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:46:54 2024

@author: HP
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.templates = self.load_templates(['chips1.jpg', 'can1.jpg', 'bottle1.jpg'])
        self.imagePath = None  # Store the path of the loaded image
        self.initUI()

    def load_templates(self, filenames):
        templates = {}
        for filename in filenames:
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                templates[filename.split('.')[0].capitalize()] = image
            else:
                print(f"Failed to load {filename}")
        return templates

    def initUI(self):
        self.setWindowTitle('Item Detector')
        self.setGeometry(300, 300, 1000, 500)

        mainLayout = QVBoxLayout()
        imageLayout = QHBoxLayout()  # To arrange images side by side

        self.labelImageInput = QLabel(self)
        self.labelImageInput.setAlignment(Qt.AlignCenter)
        self.labelImageOutput = QLabel(self)
        self.labelImageOutput.setAlignment(Qt.AlignCenter)

        btnLoad = QPushButton('Load Image', self)
        btnLoad.clicked.connect(self.openImage)

        btnDetect = QPushButton('Detect Items', self)
        btnDetect.clicked.connect(self.processImage)
        
        btnLoad.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 4px 2px;
                transition-duration: 0.4s;
                cursor: pointer;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
                border: 2px solid #4CAF50;
            }
        """)

        btnDetect.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 4px 2px;
                transition-duration: 0.4s;
                cursor: pointer;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
                border: 2px solid #4CAF50;
            }
        """)

        mainLayout.addWidget(btnLoad)
        mainLayout.addWidget(btnDetect)
        imageLayout.addWidget(self.labelImageInput)
        imageLayout.addWidget(self.labelImageOutput)
        mainLayout.addLayout(imageLayout)
        self.setLayout(mainLayout)

    def openImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if filePath:
            self.imagePath = filePath
            self.displayImage(filePath, self.labelImageInput)

    def displayImage(self, filePath, label):
        pixmap = QPixmap(filePath)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def processImage(self):
        if not self.imagePath:
            print("No image loaded.")
            return
        
        image = cv2.imread(self.imagePath)
        if image is None:
            print("Error: Image not found.")
            return
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 30:  # Consider minimum size to avoid noise
                roi = gray[y:y+h, x:x+w]
                label, score = self.identify_item(roi)
                color = (0, 255, 0) if label else (0, 0, 255)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        self.displayProcessed(image)

    def identify_item(self, roi):
        best_label = None
        best_score = 0
        for label, template in self.templates.items():
            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_label = label
        return best_label, best_score if best_label else ("Unknown", 0)

    def displayProcessed(self, img):
        # Convert the image to Qt format and display it
        height, width = img.shape[:2]
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        self.labelImageOutput.setPixmap(pixmap)
        self.labelImageOutput.setScaledContents(True)
        
        # Save the output image
        output_path = 'output_image.jpg'
        cv2.imwrite(output_path, img)
        print(f"Output image saved as {output_path}")

app = QApplication(sys.argv)
ex = ImageApp()
ex.show()
sys.exit(app.exec_())
