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
from PyQt5.QtGui import QIntValidator

from PyQt5.QtWidgets import QApplication, QWidget, QToolBar, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel, \
    QLabel, QMessageBox, QMainWindow, QStyle, QFileDialog, QSlider
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QUrl, QThread
from PyQt5 import uic

from ultralytics import YOLO
import random


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = None
        self.timer = QtCore.QTimer()
        self.paused = False  # Flag to track whether the video is paused or not

    def start_recording(self, filename):
        self.camera = cv2.VideoCapture(filename)
        self.timer.timeout.connect(self.read_frame)
        self.timer.start(0)
        print('run_button.clicked')

    def read_frame(self):
        if not self.paused:
            read, data = self.camera.read()
            if read:
                self.image_data.emit(data)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)

    def update_frame_based_on_timestamp(self, timestamp):
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, timestamp)  # Set frame position
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
        self.paint_label.setGeometry(0, 0, 800, 800)
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

        self.paused = False  # Flag to track whether the video is paused or not
        # Flag to indicate if video is restarted from frame 0
        self.video_restarted = False
        self.total_frames = 0

        # TODO: set video port
        self.record_video = RecordVideo()
        image_data_slot = self.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        self.setup_controll()

    def setup_controll(self):
        self.run_button.clicked.connect(self.start_recording)
        self.pause_button.clicked.connect(self.toggle_pause_resume)
        #horizontal slider for frame selection
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)  # Adjust the maximum value according to your video length
        self.frame_slider.setValue(0)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.valueChanged.connect(self.slider_value_changed)

        # setting geometry(size)
        #self.frame_slider.setFixedWidth(550)  # Adjust the width according to your preference
        self.run_button.setFixedWidth(130)
        self.pause_button.setFixedWidth(100)
        self.frame_jump_edit.setFixedWidth(150)
        self.jump_button.setFixedWidth(100)

        # Connect the button's clicked signal to a custom slot
        self.jump_button.clicked.connect(self.jump_to_frame)
        # Restrict jump_edit QLineEdit to accept only integer values
        int_validator = QIntValidator()
        self.frame_jump_edit.setValidator(int_validator)

    def start_recording(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        video_name = os.path.splitext(os.path.basename(filename))[0]

        if filename:
            self.video_restarted = True
            self.bbox_excel_path = f'{self.dir_path}\\bbox_save_files\\{video_name}_bounding_boxes_with_time.xlsx'

            self.total_frames = self.get_video_length(filename)
            self.frame_slider.setMaximum(self.total_frames)
            self.frame_slider.setValue(0)
            self.record_video.start_recording(filename)
            self.record_video.resume()
            self.pause_button.setText("Pause")
            print(f'self.total_frames: {self.total_frames}')

    def slider_value_changed(self, value):
        # Update the frame shown on the GUI according to the slider value
        self.timestamp = value
        # Call the method to update the frame based on the new timestamp
        self.record_video.update_frame_based_on_timestamp(self.timestamp)

    # Detect circles/human head in the image and draw them
    def image_data_slot(self, image_data):
        try:

            self.label_frame.setText(f'frame: {self.timestamp}/{self.total_frames}')
            # Check if the bounding box Excel file exists
            if not os.path.exists(self.bbox_excel_path):
                print(self.bbox_excel_path)
                raise FileNotFoundError("Bounding box Excel file not found.")

            # Read bounding box data from Excel with the first column as index
            df = pd.read_excel(self.bbox_excel_path, index_col=0)

            # Reset timestamp if video is restarted from frame 0
            if self.video_restarted:
                self.timestamp = df.index.min()
                self.video_restarted = False  # Reset the flag

            if self.timestamp in df.index:  # Check if timestamp exists in the DataFrame
                print(f'timestamp:　{self.timestamp}')
                print(df.loc[self.timestamp].values.tolist())
                print(type(df.loc[self.timestamp].values.tolist()))
                bboxes = df.loc[self.timestamp].values.tolist()
                if isinstance(bboxes[0], float):
                    bboxes = [bboxes]
                frame_with_bboxes = self.draw_bounding_boxes(image_data.copy(),
                                                             bboxes)  # Draw bounding boxes on the frame
                # show bbox position information
                bbox_list = df.loc[self.timestamp].values.tolist()
                if isinstance(bbox_list[0], float):
                    bbox_list = [bbox_list]
                len_bbox_list = len(bbox_list)
                print(f'len_bbox_list: {len_bbox_list}')
                self.label_count.setText(f'{len_bbox_list} human head(s) detected')
                # Convert list of lists to a string with newline characters ##bug
                text = '\n'.join([str(sublist[:-1] + [round(sublist[-1], 2)]) for sublist in bbox_list])

                self.label_set.setText('[x1, y1, x2, y2, score]')
                self.label_bbox.setText(text)

            else:
                frame_with_bboxes = image_data
                # reset text
                self.label_count.setText('0 human head detected')
                self.label_set.setText('')
                self.label_bbox.setText('')
                # cap_out.write(image_data)

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
            # reset size
            # if self.image.size() != self.size():
            #    self.setFixedSize(self.image.size())

            self.update()

        except Exception as e:
            print(f"Error in human head detection: {e}")

    # Function to draw bounding boxes on the frame
    def draw_bounding_boxes(self, frame, bounding_boxes):
        print(f'bounding_boxes: {bounding_boxes}')
        print(type(bounding_boxes[0]))

        for bbox in bounding_boxes:
            x1, y1, x2, y2, score = bbox  # Extracting coordinates
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Draw bounding box
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

    def toggle_pause_resume(self):
        self.record_video.paused = not self.record_video.paused
        if self.record_video.paused:
            self.record_video.pause()
            self.pause_button.setText("Resume")
        else:
            self.record_video.resume()
            self.pause_button.setText("Pause")

    def get_video_length(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file is opened successfully
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return None

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Release the video capture object
        cap.release()

        return total_frames

    def jump_to_frame(self):
        try:
            # Retrieve the text from the QLineEdit and convert it to an integer
            frame_number = int(self.frame_jump_edit.text())
            if frame_number <= self.total_frames:
                # Set the value of the QSlider to the frame number
                self.frame_slider.setValue(frame_number)
        except ValueError:
            print("Invalid frame number entered.")


if __name__ == '__main__':
    # script_dir = path.dirname(path.realpath(__file__))
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())