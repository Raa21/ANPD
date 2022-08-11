import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import time

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        h, w = None, None
        with open("C:/Users/rajen/Downloads/Compressed/yolo-coco-data/coco.names") as f:
            labels = [line.strip() for line in f]

        network = cv2.dnn.readNetFromDarknet("C:/Users/rajen/Downloads/Compressed/yolo-coco-data/yolov3.cfg",
                                    "C:/Users/rajen/Downloads/Compressed/yolo-coco-data/yolov3.weights")

        layers_names_all = network.getLayerNames()
        layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
        probability_minimum = 0.5
        threshold = 0.3
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


        while self.ThreadActive:
            ret, frame = Capture.read()
            if w is None or h is None:
                h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
            network.setInput(blob)  # setting blob as input to the network
            start = time.time()
            output_from_network = network.forward(layers_names_output)
            end = time.time()
            bounding_boxes = []
            confidences = []
            class_numbers = []
            for result in output_from_network:
        # Going through all detections from current output layer
                for detected_objects in result:
                    # Getting 80 classes' probabilities for current detected object
                    scores = detected_objects[5:]
                    # Getting index of the class with the maximum value of probability
                    class_current = np.argmax(scores)
                    # Getting value of probability for defined class
                    confidence_current = scores[class_current]

                if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial frame size
                # YOLO data format keeps coordinates for center of bounding box
                # and its current width and height
                # That is why we can just multiply them elementwise
                # to the width and height
                # of the original frame and in this way get coordinates for center
                # of bounding box, its width and height for original frame
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Now, from YOLO data format, we can get top left corner coordinates
                    # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min,
                                        int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

            results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

            if len(results) > 0:
        # Going through indexes of results
                for i in results.flatten():
                    # Getting current bounding box coordinates,
                    # its width and height
                    x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                    box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                    # Preparing colour for current bounding box
                    # and converting from numpy array to list
                    colour_box_current = colours[class_numbers[i]].tolist()

                    # # # Check point
                    # print(type(colour_box_current))  # <class 'list'>
                    # print(colour_box_current)  # [172 , 10, 127]

                    # Drawing bounding box on the original current frame
                    cv2.rectangle(frame, (x_min, y_min),
                                (x_min + box_width, y_min + box_height),
                                colour_box_current, 2)

                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                        confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(500, 500, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())