import cv2
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QToolBar, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel, \
    QLabel, QMessageBox, QMainWindow, QStyle, QFileDialog
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QUrl, QThread
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5 import uic
import sys
import face_recognition
import os,sys
import cv2
import numpy as np
import math
import face_rec


class MainWindow(QMainWindow):
    file = ''
    def __init__(self):
        super(MainWindow,self).__init__()
        # load ui (videowidget object or so)
        uic.loadUi("mainwindow.ui", self)
        self.setFixedSize(QSize(1600,600))
        self.setWindowTitle("Face Recognition GUI")
        #set color
        p = self.palette()
        p.setColor(QPalette.Window, Qt.white)
        self.setPalette(p)

        self.setup_controll()


    def setup_controll(self):
        # create media player object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)


        ## signal
        # open file button
        self.openBtn.clicked.connect(self.open_file)

        # button for playing
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)

        #slider
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)

        #face check
        self.face_chkbox.setChecked(False)
        self.face_chkbox.clicked.connect(self.onCheck_face_chkbox_Click)


        # QMediaPlayer
        self.mediaPlayer.setVideoOutput(self.videowidget)

        # media player signals
        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        #cancel button
        self.cancelBtn.clicked.connect(self.CancelFeed)
        self.startBtn.clicked.connect(self.StartFeed)
        self.startBtn.setEnabled(False)

        #thread
        self.Worker1 = Worker1()
        self.Worker1.encode_faces()

        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)



    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def on_clicked(msg):
        message = QMessageBox()
        message.setText("Hello world "+msg)
        message.exec_()
    # Create a Qt widget, which will be our window.

    def the_button_was_clicked(self):
        print("Clicked.")

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def mediastate_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)

            )

        else:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)

            )
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")

        if filename != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playBtn.setEnabled(True)
            self.startBtn.setEnabled(True)
            self.Worker1.getfilename(filename)
    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()

        else:
            self.mediaPlayer.play()
    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def onCheck_face_chkbox_Click(self):
        if self.face_chkbox.isChecked() and self.file != '':
            print('face recognition activate')
            print(self.file)
            fr = face_rec.FaceRecognition()
            fr.run_recognition(self.file)

        else:
            print('face recognition deactivate')
    def CancelFeed(self):
        self.Worker1.stop()

    def StartFeed(self):
        self.Worker1.start()

    #add element

#thread is going to handle retrieving image from webcam and convert to a format that pyQt can understand
#work1 makes a connection with camera(video)
class Worker1(QThread):
    filename = []

    #for face_recognition
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True


    #define the signal that thread is going to emit
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        # create a video capture
        Capture = cv2.VideoCapture(self.filename)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if(ret):
                small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

                Image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all faces in the current frame
                self.face_locations = face_recognition.face_locations(Image)
                # do encodings
                self.face_encodings = face_recognition.face_encodings(Image, self.face_locations)

                self.face_names = []
                # for face_encoding in self.face_encodings:
                #     # check if there are any matches
                #     ####
                #     matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                #     # default values
                #     name = 'Unknown'
                #     confidence = 'Unknown'
                #
                #     face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                #     best_match_index = np.argmin(face_distances)
                #
                #     # giving recognized label
                #     if matches[best_match_index]:
                #         name = self.known_face_names[best_match_index]
                #         confidence = self.face_confidence(face_distances[best_match_index])
                #
                #     self.face_names.append(f'{name} ({confidence})')



                FlippedImage = cv2.flip(Image, 1)
                #convert frame to QImage
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def stop(self):
        self.ThreadActive = False
        self.quit()

    def getfilename(self,file):
        self.filename = file

    def face_confidence(face_distance, face_match_threshold=0.6):
        range = (1.0 - face_match_threshold)
        # calculate the percetage for the accuracy
        linear_val = (1.0 - face_distance) / (range * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + "%"
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + "%"

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            ### no [0]
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            #append known names in faces
            self.known_face_names.append(os.path.splitext(image)[0])

        print(self.known_face_names)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()  # IMPORTANT!!!!! Windows are hidden by default.
    # Start the event loop
    app.exec_()