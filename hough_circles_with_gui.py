import sys
from os import path

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPalette
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

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


class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, haar_cascade_filepath, parent=None):
        super().__init__(parent)
        self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    def detect_circles(self, image: np.ndarray):
        # haarclassifiers work better in black and white
        output = image.copy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.medianBlur(gray_image, 5)
        #gray_image = cv2.equalizeHist(gray_image)
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=0, maxRadius=0)
        detected_circles = np.uint16(np.around(circles))

        faces = self.classifier.detectMultiScale(gray_image,
                                                 scaleFactor=1.3,
                                                 minNeighbors=4,
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 minSize=self._min_size)

        return detected_circles

    def image_data_slot(self, image_data):
        faces = self.detect_circles(image_data)
        #draw the circles
        for (x, y, r) in faces[0, :]:
            cv2.circle(image_data, (x, y), r, (0, 0, 0), 3)
            cv2.circle(image_data, (x, y), 2, (0, 255, 255), 3)

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)


class MainWidget(QtWidgets.QWidget):
    def __init__(self, haarcascade_filepath, parent=None):
        super().__init__(parent)
        fp = haarcascade_filepath
        self.face_detection_widget = FaceDetectionWidget(fp)

        # TODO: set video port
        self.record_video = RecordVideo()

        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.face_detection_widget)
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
    def __init__(self, haarcascade_filepath, parent=None):
        super().__init__(parent)
        uic.loadUi("mainwindow.ui", self)
        self.setFixedSize(QSize(1300, 700))
        self.setWindowTitle("Face Recognition GUI")
        fp = haarcascade_filepath
        p = self.palette()
        p.setColor(QPalette.Window, Qt.white)
        self.setPalette(p)
        self.main_widget = MainWidget(haarcascade_filepath)
        self.setCentralWidget(self.main_widget)

        #self.setup_controll()

    #def setup_controll(self):



def main(haar_cascade_filepath):
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(haar_cascade_filepath)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cascade_filepath = path.join(script_dir,
                                 '..',
                                 'pythonProject_FaceDetection',
                                 'haarcascade_frontalface_default.xml')

    cascade_filepath = path.abspath(cascade_filepath)
    main(cascade_filepath)