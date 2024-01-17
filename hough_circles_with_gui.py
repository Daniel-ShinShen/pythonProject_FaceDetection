import sys
from os import path

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPalette

from PyQt5.QtWidgets import QApplication, QWidget, QToolBar, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel, \
    QLabel, QMessageBox, QMainWindow, QStyle, QFileDialog
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QUrl, QThread
from PyQt5 import uic
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
        self.filename = file
        self.camera = cv2.VideoCapture(self.filename)
        print(self.filename)


class CircleDetectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

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

    def image_data_slot(self, image_data):
        circles = self.detect_circles(image_data)
        if circles is not None:
            #draw the circles
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(image_data, center, radius, (0, 0, 0), 3)
                cv2.circle(image_data, center, 2, (0, 255, 255), 3)

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

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
        painter.drawImage(0, 0, self.image)


class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.circle_detection_widget = CircleDetectionWidget()

        # TODO: set video port
        self.record_video = RecordVideo()

        image_data_slot = self.circle_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.circle_detection_widget)
        self.run_button = QtWidgets.QPushButton('Start')
        layout.addWidget(self.run_button)

        self.run_button.clicked.connect(self.start_recording)
        self.setLayout(layout)

    def start_recording(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename:
            self.record_video.start_recording(filename)


class MainWindow(QMainWindow):
    file = ''
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("mainwindow.ui", self)
        self.setFixedSize(QSize(1300, 850))
        self.setWindowTitle("Face Recognition GUI")
        p = self.palette()
        p.setColor(QPalette.Window, Qt.white)
        self.setPalette(p)
        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)

        #self.setup_controll()

    #def setup_controll(self):



def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    main()