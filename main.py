import sys
from os import path
import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPalette, QPixmap, QImage

from PyQt5.QtWidgets import QApplication, QWidget, QToolBar, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel, \
    QLabel, QMessageBox, QMainWindow, QStyle, QFileDialog
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QUrl, QThread
from PyQt5 import uic

from ultralytics import YOLO
import random
class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = None
        self.timer = QtCore.QBasicTimer()

    def start_recording(self, filename):
        self.camera = cv2.VideoCapture(filename)
        self.timer.start(0, self)
        print('run_button.clicked')

    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)

    def getfilename(self, file):
        # Store the filename for reference
        self.filename = file
        self.camera = cv2.VideoCapture(self.filename)
        print(self.filename)



class MainWindow(QMainWindow):
    file = ''
    def __init__(self, parent=None):
        super().__init__(parent)
        # Load the UI file
        uic.loadUi("mainwindow.ui", self)

        # Set window properties
        self.setWindowTitle("Face Recognition GUI")
        # self.setFixedSize(QSize(800, 850))
        p = self.palette()
        p.setColor(QPalette.Window, Qt.white)
        self.setPalette(p)

        # Create an image label for displaying video frames
        self.paint_label.setGeometry(0, 0, 600, 600)
        self.image = QtGui.QImage(self.paint_label.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.paint_label.setPixmap(QPixmap.fromImage(self.image))
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        # create a list of random color for tracking
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]


        # TODO: set video port
        self.record_video = RecordVideo()
        image_data_slot = self.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        self.setup_controll()
    def setup_controll(self):
        self.run_button.clicked.connect(self.start_recording)
    def start_recording(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename:
            self.record_video.start_recording(filename)

    # Detect circles/human head in the image and draw them
    def image_data_slot(self, image_data):
        try:
            model = YOLO("best_20epoch.pt")


            results = model(image_data)
            print(results)

            for result in results:  # only one object inside results
                detections = []
                for r in result.boxes.data.tolist():
                    print(r)
                    x1, y1, x2, y2, score, class_id = r  # unwrap the information
                    # read
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    class_id = int(class_id)
                    detections.append([x1, y1, x2, y2, score])

                    cv2.rectangle(image_data, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)#(self.colors[100 % len(self.colors)])
            """""
            circles = self.detect_circles(image_data)
            if circles is not None:
                # Draw the circles
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    cv2.circle(image_data, center, radius, (0, 0, 0), 3)
                    cv2.circle(image_data, center, 2, (0, 255, 255), 3)
            """""
            self.image = self.get_qimage(image_data)
            if self.image.size() != self.size():
                self.setFixedSize(self.image.size())

            self.update()

        except Exception as e:
            print(f"Error in circle detection: {e}")

    # Resize and convert the image to a QImage
    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape

        # Set the desired width for resizing
        target_width = 800  # Adjust this value according to your preference
        # Calculate the corresponding height to maintain aspect ratio
        target_height = int(height * (target_width / width))
        # Resize the image
        resized_image = cv2.resize(image, (target_width, target_height))

        bytesPerLine = 3 * target_width
        QImage = QtGui.QImage

        image = QImage(resized_image.data,
                       target_width,
                       target_height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        #painter.drawImage(0, 0, self.image)
        self.paint_label.setPixmap(QPixmap.fromImage(self.image))
    def detect_circles(self, image: np.ndarray):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply median blur to reduce noise
        gray_image = cv2.medianBlur(gray_image, 5)
        edges = cv2.Canny(gray_image, 100, 350)
        # Use Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10,
                                  param1=50, param2=15, minRadius=14, maxRadius=30)
        if circles is not None:
            detected_circles = np.uint16(np.around(circles))
            return detected_circles
        else:
            return []



if __name__ == '__main__':
    #script_dir = path.dirname(path.realpath(__file__))
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
