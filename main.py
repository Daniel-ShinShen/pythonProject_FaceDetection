import os
import sys
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

# count
from collections import Counter
from collections import deque
import math

import time

# tracking
# from deep_sort.utils.parser import get_config
#from deep_sort.deep_sort import DeepSort
#from deep_sort.sort.tracker import Tracker

import threading
from ultralytics import YOLO
import random


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)
    time_count_signal = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.time_count = 0
        self.camera = None
        self.timer = QtCore.QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.paused = False  # Flag to track whether the video is paused or not

    def start_recording(self, filename):
        self.time_count = 0
        self.camera = cv2.VideoCapture(filename)
        self.timer.timeout.connect(self.read_frame)  # 設定定時要執行的 function
        self.timer.start(0)  # 啟用定時器，設定間隔時間為 0 毫秒
        print('run_button.clicked')

    def read_frame(self):
        if not self.paused:
            read, data = self.camera.read()
            if read:
                self.time_count += 1
                self.image_data.emit(data)
                self.time_count_signal.emit(self.time_count)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def update_frame_based_on_timestamp(self, timestamp):
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, timestamp)  # Set frame position
        read, data = self.camera.read()
        if read:
            self.time_count = timestamp
            self.image_data.emit(data)
            self.time_count_signal.emit(self.time_count)


class WorkerThread(QtCore.QThread):
    update_slider = QtCore.pyqtSignal()  # Signal to update slider position

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = None
        self.mutex = QtCore.QMutex()  # Mutex to control access to paused flag
        self.paused = False

    def run(self):
        def work():
            print("working from :" + str(threading.get_ident()))
            QThread.sleep(5) # pauses the execution of the thread where it's called for 5 seconds

        # This method will be executed in a separate thread
        self.timer = QtCore.QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.send_signal)
        self.timer.timeout.connect(work)
        self.timer.start(200)
        print("thread started from :" + str(threading.get_ident()))
        self.exec_()

    def pause(self):
        with QtCore.QMutexLocker(self.mutex):
            self.paused = True
            #self.timer.stop()

    def resume(self):
        with QtCore.QMutexLocker(self.mutex):
            self.paused = False
            #self.timer.start(200)

    def stop(self):
        self.timer.stop()  # 停止定時器
        self.terminate()

    def send_signal(self):
        value = 10
        if self.paused is False:
            self.update_slider.emit()
        print("send signal")


class MainWindow(QMainWindow):
    file = ''

    def __init__(self, parent=None):
        super().__init__(parent)
        # Load the UI file
        uic.loadUi("mainwindow.ui", self)

        # Set window properties
        self.setWindowTitle("Human Head Detection GUI")
        self.setWindowIcon(QtGui.QIcon('logo.png'))
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

        self.video_name = ''
        # setting timestamp and some excel related stuff
        self.timestamp = 0
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.filename = ''
        self.bbox_excel_path = ''
        self.tracking_excel_path = ''
        self.counting_excel_path = ''
        #self.video_out_path = ''

        self.paused = False  # Flag to track whether the video is paused or not
        # Flag to indicate if video is restarted from frame 0
        self.video_restarted = False
        self.total_frames = 0
        self.df = None
        self.df_tracking = None
        self.df_counting = None

        # initialize trajectory data
        self.center_points = []

        # tracking/people counting data
        self.class_names = ['human_head', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.colors = np.array([
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0]
        ])

        self.paths = {}
        self.total_counter = 0
        self.down_count = 0
        self.up_count = 0
        self.counted_data_down = []
        self.counted_data_up = []
        self.last_track_id = -1
        self.class_counter = Counter()  # store counts of each detected class
        self.already_counted = deque(maxlen=15)  # temporary memory for storing counted IDs
        self.line = []

        self.label_person_count.setText('People Count:')
        self.label_person_up.setText('People Upward:')
        self.label_person_down.setText('People Downward:')

        #deep_sort_weights = f'{self.dir_path}\\deep_sort\\deep\\checkpoint\\ckpt.t7'
        #self.tracker = DeepSort(model_path=deep_sort_weights, max_age=3)

        # thread
        self.worker = WorkerThread(self)

        # TODO: set video port
        self.record_video = RecordVideo()
        self.record_video.image_data.connect(self.image_data_slot)
        # frame counting
        self.record_video.time_count_signal.connect(self.time_count_slot)

        self.worker.update_slider.connect(self.update_slider_position)#####

        self.setup_controll()

    def setup_controll(self):
        self.run_button.clicked.connect(self.start_recording)
        self.pause_button.clicked.connect(self.toggle_pause_resume)
        # horizontal slider for frame selection
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)  # Adjust the maximum value according to your video length
        self.frame_slider.setValue(0)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.valueChanged.connect(self.slider_value_changed)

        # Connect the combobox's signal to handle the selection
        self.excel_combobox.currentIndexChanged.connect(self.handle_combobox_selection)

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

        # trajectory checkBox
        self.trajectory_checkBox.clicked.connect(self.on_trajectory_checkBox_click)
        # people count checkBox
        self.peoplecount_checkBox.clicked.connect(self.on_people_count_checkBox_click)
        #bbox_checkBox
        self.bbox_checkBox.setChecked(True)
        self.bbox_checkBox.clicked.connect(self.on_bbox_checkBox_click)

    def start_recording(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        self.video_name = os.path.splitext(os.path.basename(filename))[0]

        if filename:
            self.video_restarted = True
            self.bbox_excel_path = (f'{self.dir_path}\\bbox_save_files'
                                    f'\\{self.video_name}_bounding_boxes_with_time_v8.xlsx')
            self.tracking_excel_path = (f'{self.dir_path}\\tracking_save_files'
                                        f'\\{self.video_name}_tracking_results_with_time.xlsx')
            self.counting_excel_path = (f'{self.dir_path}\\counting_save_files'
                                        f'\\{self.video_name}_people_counting_with_time.xlsx')

            # combobox
            # Clear the current items in the combobox
            if self.excel_combobox.count() > 0:
                self.excel_combobox.blockSignals(True)
                self.excel_combobox.clear()

            self.populate_combobox()

            self.total_frames = self.get_video_length(filename)
            self.frame_slider.setMaximum(self.total_frames-1)
            self.frame_slider.setValue(0)
            self.timestamp = 0
            self.record_video.start_recording(filename)
            print(f'self.total_frames: {self.total_frames}')

            # reset trajectory data
            self.center_points = []

            # start thread
            self.worker.start()

            # load data from Excel data file
            # Check if the bounding box Excel file exists
            if not os.path.exists(self.bbox_excel_path):
                print(self.bbox_excel_path)
                raise FileNotFoundError("Bounding box Excel file not found.")

            # Read bounding box data from Excel with the first column as index
            self.df = pd.read_excel(self.bbox_excel_path, index_col=0)

            # Check if the Tracking Excel file exists
            if not os.path.exists(self.tracking_excel_path):
                print(self.tracking_excel_path)
                raise FileNotFoundError("Tracking Excel file not found.")

            self.df_tracking = pd.read_excel(self.tracking_excel_path, index_col=0)

            # Check if the Counting Excel file exists
            if not os.path.exists(self.counting_excel_path):
                print(self.counting_excel_path)
                raise FileNotFoundError("Counting Excel file not found.")

            self.df_counting = pd.read_excel(self.counting_excel_path, index_col=0)

    def update_slider_position(self):  # Bottleneck
        trajectory_data = self.center_points.copy()
        if self.timestamp <= self.total_frames:
            self.frame_slider.setValue(self.timestamp)
        self.center_points = trajectory_data.copy()

    def slider_value_changed(self, value):
        # Update the frame shown on the GUI according to the slider value
        self.timestamp = value
        self.center_points = []
        # Call the method to update the frame based on the new timestamp
        self.record_video.update_frame_based_on_timestamp(self.timestamp)

    # Receive "time_count_signal" signal to increase timestamp
    def time_count_slot(self, time_count):
        self.timestamp = time_count
        #if time_count <= self.total_frames:
        #    self.frame_slider.setValue(time_count)

        # set slider position per 100 frame
        # if self.timestamp % 100 == 0:
        #    self.frame_slider.setValue(self.timestamp)
        print(f'time_count: {time_count}')

    # Receive "image_data" signal from self.record_video and process the frame(image_data)
    # Detect human head in the image and draw the bounding boxes
    def image_data_slot(self, image_data):
        try:
            bboxes_xywh = []
            conf = []
            self.label_frame.setText(f'frame: {self.timestamp}/{self.total_frames - 1}')

            # Reset timestamp if video is restarted from frame 0
            if self.video_restarted:
                self.timestamp = 0
                self.video_restarted = False  # Reset the flag

            # draw line where people crossing (used for counting)
            if self.peoplecount_checkBox.isChecked():
                self.line = [(0, int(0.5 * image_data.shape[0])),
                        (int(image_data.shape[1]), int(0.5 * image_data.shape[0]))]
                line_y = int(0.5 * image_data.shape[0])
                cv2.line(image_data, self.line[0], self.line[1], (255, 255, 0), 10)

            if self.timestamp in self.df.index:  # Check if timestamp exists in the DataFrame(storing bounding boxes)
                print(self.df.loc[self.timestamp].values.tolist())
                bboxes = self.df.loc[self.timestamp].values.tolist()

                if isinstance(bboxes[0], float):
                    bboxes = [bboxes]
                # Draw bounding boxes on the frame # always do this
                frame_with_bboxes = self.draw_bounding_boxes(image_data.copy(),
                                                             bboxes)
                # Draw trajectory
                if self.trajectory_checkBox.isChecked():
                    frame_with_bboxes = self.draw_trajectory(frame_with_bboxes)

                # Convert bboxes_xywh to numpy array
                #bboxes_xywh = np.array(bboxes_xywh, dtype=float)
                #conf = np.array(conf, dtype=float)

                # Show bbox position information
                bbox_list = self.df.loc[self.timestamp].values.tolist()
                if isinstance(bbox_list[0], float):
                    bbox_list = [bbox_list]
                len_bbox_list = len(bbox_list)
                print(f'len_bbox_list: {len_bbox_list}')
                self.label_count.setText(f'{len_bbox_list} human head(s) detected')
                # Convert list of lists to a string with newline characters ##bug
                text = '\n'.join([str(sublist[:-1] + [round(sublist[-1], 2)]) for sublist in bbox_list])
                self.label_bbox.setText(text)

            else:
                frame_with_bboxes = image_data

                if self.trajectory_checkBox.isChecked():
                    # Draw trajectory
                    frame_with_bboxes = self.draw_trajectory(frame_with_bboxes)
                # reset text
                self.label_count.setText('0 human head detected')
                self.label_set.setText('')
                self.label_bbox.setText('')

            # Check if timestamp exists in the DataFrame(storing tracking data)
            if self.timestamp in self.df_tracking.index:
                tracks = self.df_tracking.loc[self.timestamp].values.tolist()
                # print('tracks: ', tracks)
                if isinstance(tracks[0], float):
                    tracks = [tracks]
                # Draw tracking boxes on the frame
                if self.peoplecount_checkBox.isChecked():
                    frame_with_bboxes = self.draw_tracking_boxes(frame_with_bboxes.copy(), tracks)
            else:
                frame_with_bboxes = frame_with_bboxes

            # Check if timestamp exists in the DataFrame(storing counting data)
            if self.timestamp in self.df_counting.index:
                counting = self.df_counting.loc[self.timestamp].values.tolist()
                print('counting: ', counting)

                if self.peoplecount_checkBox.isChecked():
                    self.label_person_count.setText(f'People Count: {counting[0]}')
                    self.label_person_up.setText(f'People Upward: {counting[1]}')
                    self.label_person_down.setText(f'People Downward: {counting[2]}')
                else:
                    self.label_person_count.setText('People Count:')
                    self.label_person_up.setText('People Upward:')
                    self.label_person_down.setText('People Downward:')

            self.label_set.setText('[x1, y1, x2, y2, score]')
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
            #print(f'timestamp:　{self.timestamp}')
            #self.timestamp = self.timestamp + 1  # used before creating time_count_signal to set timestamp value
            self.image = self.get_qimage(frame_with_bboxes)

            # reset size
            # if self.image.size() != self.size():
            #    self.setFixedSize(self.image.size())
            self.update()

        except Exception as e:
            print(f"Error in image_data_slot: {e}")

    # Function to draw bounding boxes on the frame
    def draw_bounding_boxes(self, frame, bounding_boxes):
        print(f'bounding_boxes: {bounding_boxes}')

        frame_processed = frame.copy()

        for bbox in bounding_boxes:
            x1, y1, x2, y2, score = bbox  # Extracting coordinates
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # save trajectory data
            self.center_points.append((cx, cy))
            # show bounding boxes
            if self.bbox_checkBox.isChecked():
                frame_processed = cv2.rectangle(frame_processed, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Draw bounding box
            else:
                frame_processed = frame_processed

        # Remove the first n elements from center_points after it reaches m elements
        m = 100  # Define the threshold for m
        n = 10  # Define the number of elements to remove
        if len(self.center_points) > m:
            self.center_points = self.center_points[n:]

        return frame_processed

    def draw_tracking_boxes(self, frame, tracks):
        print(f'tracks: {tracks}')

        for track in tracks:
            x1, y1, x2, y2, track_id = track  # Extracting coordinates
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])  # get midpoint respective to botton-left

            color = self.compute_color_for_labels(track_id)
            class_name = self.class_names[int(0)]  # all detected objects are human head

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Draw bounding box
            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, text_color, 2, cv2.LINE_AA)
            if track_id not in self.paths:
                self.paths[track_id] = deque(maxlen=2)
                total_track = track_id

            self.paths[track_id].append(midpoint)
            previous_midpoint = self.paths[track_id][0]
            origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])

            if (self.intersect(midpoint, previous_midpoint, self.line[0], self.line[1])
                    and track_id not in self.already_counted):
                self.class_counter[0] += 1
                # Update the person count
                self.total_counter += 1
                last_track_id = track_id

                # draw red line
                cv2.line(frame, self.line[0], self.line[1], (255, 150, 0), 10)

                self.already_counted.append(track_id)  # Set already counted for ID to true.

                angle = self.vector_angle(origin_midpoint, origin_previous_midpoint)
                if angle > 0:
                    self.up_count += 1
                if angle < 0:
                    self.down_count += 1

            if len(self.paths) > 15:
                del self.paths[list(self.paths)[0]]

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
            self.worker.pause()  # stop timer in thread
            self.pause_button.setText("Resume")
        else:
            self.record_video.resume()
            self.worker.resume()  # resume/start timer in thread
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
                # reset trajectory data
                self.center_points = []
                self.toggle_pause_resume()
                # Set the value of the QSlider to the frame number
                self.frame_slider.setValue(frame_number)
                self.toggle_pause_resume()
        except ValueError:
            print("Invalid frame number entered.")

    def populate_combobox(self):
        excel_files_versions = []
        # Get a list of available Excel files in the directory
        excel_files = [file for file in os.listdir(f'{self.dir_path}\\bbox_save_files\\') if file.endswith('.xlsx')
                       and self.video_name in file]
        for excel_file in excel_files:
            version_name = excel_file.split('_')
            excel_files_versions.append(version_name[8])
        # Populate the combobox with the filtered list of Excel files
        self.excel_combobox.addItems(excel_files_versions)
        self.excel_combobox.setCurrentText("v8.xlsx")
        self.excel_combobox.blockSignals(False)

    def handle_combobox_selection(self, index):
        # Get the selected Excel file from the combobox
        selected_excel_file = self.excel_combobox.currentText()
        # Update the bbox_excel_path variable accordingly
        self.bbox_excel_path = (f'{self.dir_path}\\bbox_save_files\\{self.video_name}_bounding_boxes_with_time'
                                f'_{selected_excel_file}')

        # reset trajectory data
        self.center_points = []

        # reloading from the new Excel file
        # Check if the bounding box Excel file exists
        if not os.path.exists(self.bbox_excel_path):
            print(self.bbox_excel_path)
            raise FileNotFoundError("Bounding box Excel file not found.")
        # Read bounding box data from Excel with the first column as index
        self.df = pd.read_excel(self.bbox_excel_path, index_col=0)

        print(f'current version: {selected_excel_file}')

    def on_trajectory_checkBox_click(self):
        if self.trajectory_checkBox.isChecked():
            print('trajectory checkBox is checked')

        else:
            print('trajectory checkBox is unchecked')

    def on_people_count_checkBox_click(self):
        if self.peoplecount_checkBox.isChecked():
            print('people count checkBox is checked')
            if self.timestamp in self.df_counting.index:
                counting = self.df_counting.loc[self.timestamp].values.tolist()
                print('counting: ', counting)
                self.label_person_count.setText(f'People Count: {counting[0]}')
                self.label_person_up.setText(f'People Upward: {counting[1]}')
                self.label_person_down.setText(f'People Downward: {counting[2]}')
        else:
            print('people count checkBox is unchecked')
            self.label_person_count.setText('People Count:')
            self.label_person_up.setText('People Upward:')
            self.label_person_down.setText('People Downward:')

    def on_bbox_checkBox_click(self):
        if self.bbox_checkBox.isChecked():
            print('bbox checkBox is checked')
        else:
            print('bbox checkBox is unchecked')


    def draw_trajectory(self, frame):

        for pt in self.center_points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        return frame

    def intersect(self, a, b, c, d):
        return self.ccw(a, c, d) != self.ccw(b, c, d) and self.ccw(a, b, c) != self.ccw(a, b, d)

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def vector_angle(self, midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))

    def get_color(self, c, x, max_value):
        ratio = (x / max_value) * 5
        i = math.floor(ratio)
        j = math.ceil(ratio)
        ratio -= i
        r = (1 - ratio) * self.colors[i][c] + ratio * self.colors[j][c]
        return r

    def compute_color_for_labels(self, class_id, class_total=80):
        offset = (class_id + 0) * 123457 % class_total
        red = self.get_color(2, offset, class_total)
        green = self.get_color(1, offset, class_total)
        blue = self.get_color(0, offset, class_total)
        return int(red * 256), int(green * 256), int(blue * 256)

    def closeEvent(self, event):
        # Stop the worker thread before closing the application
        self.worker.stop()
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())