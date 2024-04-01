import os
import sys
from os import path
import cv2
import numpy as np
import pandas as pd

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
        print(f'filename: {self.filename}')


class MainWindow(QMainWindow):
    file = ''

    def __init__(self, parent=None):
        super().__init__(parent)
        # Load the UI file
        uic.loadUi("mainwindow.ui", self)

        # Set window properties
        self.setWindowTitle("Human Head Detection GUI")
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

        self.video_name = os.path.splitext(os.path.basename(self.file))[0]
        # setting timestamp and some excel related stuff
        self.timestamp = 0
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.filename = ''
        self.bbox_excel_path = ''
        self.video_out_path = ''

        # TODO: set video port
        self.record_video = RecordVideo()
        image_data_slot = self.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        self.setup_controll()

    def setup_controll(self):
        self.run_button.clicked.connect(self.start_recording)

    def start_recording(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        video_name = os.path.splitext(os.path.basename(filename))[0]
        self.bbox_excel_path = f'{self.dir_path}\\bbox_save_files\\{video_name}_bounding_boxes_with_time.xlsx'
        self.video_out_path = f'{self.dir_path}\\loading_excel_export_file\\{video_name}_loading_excel_out_0401.mp4'
        if filename:
            self.record_video.start_recording(filename)
            print(self.video_name)


    # Detect circles/human head in the image and draw them
    def image_data_slot(self, image_data):
        try:


            # Check if the bounding box Excel file exists
            if not os.path.exists(self.bbox_excel_path):
                print(self.bbox_excel_path)
                raise FileNotFoundError("Bounding box Excel file not found.")


            # Read bounding box data from Excel with the first column as index
            df = pd.read_excel(self.bbox_excel_path, index_col=0)
            #cap = cv2.VideoCapture(self.filename)
            #cap_out = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                                      #(image_data.shape[1], image_data.shape[0]))
            if self.timestamp in df.index:  # Check if timestamp exists in the DataFrame
                print(f'timestamp:　{self.timestamp}')
                print(df.loc[self.timestamp].values.tolist())
                print(type(df.loc[self.timestamp].values.tolist()))
                bboxes = df.loc[self.timestamp].values.tolist()
                frame_with_bboxes = self.draw_bounding_boxes(image_data.copy(), bboxes)  # Draw bounding boxes on the frame
                #cap_out.write(frame_with_bboxes)
                # for bbox in df.loc[timestamp].values.tolist():
                # frame_with_bboxes = draw_bounding_boxes(frame.copy(), [bbox])  # Draw bounding boxes on the frame
                # cap_out.write(frame_with_bboxes)
            else:
                frame_with_bboxes = image_data
                #cap_out.write(image_data)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
                return 0

            self.timestamp = self.timestamp + 1

            """""
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

                    cv2.rectangle(image_data, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255),
                                  3)  # (self.colors[100 % len(self.colors)])
            
            """""
            self.image = self.get_qimage(frame_with_bboxes)
            if self.image.size() != self.size():
                self.setFixedSize(self.image.size())

            self.update()

        except Exception as e:
            print(f"Error in human head detection: {e}")

    # Function to draw bounding boxes on the frame
    def draw_bounding_boxes(self, frame, bounding_boxes):

        if isinstance(bounding_boxes[0], float):
            bounding_boxes = [bounding_boxes]

        for bbox in bounding_boxes:
            x1, y1, x2, y2, score = bbox  # Extracting coordinates
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw bounding box
        return frame

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

        # painter.drawImage(0, 0, self.image)
        self.paint_label.setPixmap(QPixmap.fromImage(self.image))



if __name__ == '__main__':
    # script_dir = path.dirname(path.realpath(__file__))
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
