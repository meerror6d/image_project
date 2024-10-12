import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.templates = {
            'Chips': cv2.imread('chips1.jpg', cv2.IMREAD_GRAYSCALE),
            'Can': cv2.imread('can1.jpg', cv2.IMREAD_GRAYSCALE),
            'Bottle': cv2.imread('bottle1.jpg', cv2.IMREAD_GRAYSCALE)
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Item Detector')
        self.setGeometry(300, 300, 1000, 500)

        mainLayout = QVBoxLayout()
        imageLayout = QHBoxLayout()  # To arrange images side by side

        self.labelImageInput = QLabel(self)
        self.labelImageInput.setAlignment(Qt.AlignCenter)
        self.labelImageOutput = QLabel(self)
        self.labelImageOutput.setAlignment(Qt.AlignCenter)

        btnLoad = QPushButton('Take Image', self)
        btnLoad.clicked.connect(self.openImage)
        btnProcess = QPushButton('Detect Items', self)
        btnProcess.clicked.connect(self.processImage)

        # Styling the buttons
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

        btnProcess.setStyleSheet("""
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
        mainLayout.addWidget(btnProcess)
        imageLayout.addWidget(self.labelImageInput)
        imageLayout.addWidget(self.labelImageOutput)
        mainLayout.addLayout(imageLayout)
        self.setLayout(mainLayout)

    def openImage(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png)", options=options)
        if filePath:
            self.imagePath = filePath
            self.displayImage(filePath)

    def displayImage(self, filePath):
        pixmap = QPixmap(filePath)
        self.labelImageInput.setPixmap(pixmap)
        self.labelImageInput.setScaledContents(True)
        self.resize(pixmap.width(), pixmap.height())

    def processImage(self):
        if not hasattr(self, 'imagePath'):
            print("No image loaded.")
            return
        
        image = cv2.imread(self.imagePath)
        if image is None:
            print("Error: Image not found.")
            return
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray_image',image_gray)
        
        _, binary_image = cv2.threshold(image_gray, 220, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('binary image', binary_image)
        cv2.imwrite('binary_image.jpg', binary_image)
        
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output_image = image.copy()  # Use the original color image for output
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
    
        # Display the image with contours
        cv2.imshow('Contours', output_image)
        cv2.imwrite('contour image.jpg', output_image)

        min_area = 500
        item_number = 0
        color_map = {
            'Chips': (255, 0, 0),  # Blue
            'Can': (0, 255, 0),    # Green
            'Bottle': (0, 0, 255)  # Red
        }

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                roi = image_gray[y:y+h, x:x+w]
                
                roi_resized = cv2.resize(roi, (self.templates['Chips'].shape[1], self.templates['Chips'].shape[0]))
                best_match = self.identify_item(roi_resized, self.templates)
                item_number += 1
                color = color_map.get(best_match[0], (255, 255, 255))  # Default to white if no match found
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
                label_position = (x + 5, y + 15)
                cv2.putText(output_image, best_match[0], label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                window_name = f'Item {item_number}: {best_match[0]}'
                cv2.imshow(window_name, roi_resized)
                cv2.imwrite('roi.jpg', roi_resized)
                print(f'Item {item_number}: {best_match[0]} with score {best_match[1]}')
                #cv2.imwrite('output',output_image)
                save_path=''

        self.displayProcessed(output_image,save_path)

    def displayProcessed(self, img,save_path):
        height, width = img.shape[:2]
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        self.labelImageOutput.setPixmap(pixmap)
        self.labelImageOutput.setScaledContents(True)
        self.resize(pixmap.width(), pixmap.height())
        
        if save_path:
            cv2.imwrite('save_path', img)

    def match_template(self, image, template):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(image, None)
        if des1 is None or des2 is None:
            return 0
        
        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
        matches = flann.knnMatch(des1, des2, k=2)
        return len([m for m, n in matches if m.distance < 0.7 * n.distance])

    def identify_item(self, roi, templates):
        scores = {label: self.match_template(roi, template) for label, template in templates.items()}
        return max(scores.items(), key=lambda item: item[1])

app = QApplication(sys.argv)
ex = ImageApp()
ex.show()
sys.exit(app.exec_())
