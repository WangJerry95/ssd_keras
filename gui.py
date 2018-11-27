import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from detect import Ui_MainWindow


import numpy as np
import cv2

from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from keras.optimizers import SGD
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels


PROBABILITY_THRESHOLD = 0.99

class App(QMainWindow):

    def __init__(self):
        super(App, self).__init__()
        self.ui =Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.actionOpen.triggered.connect(self.openFileNameDialog)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionBuild_model.triggered.connect(self.load_model)
        self.ui.actionDetect.triggered.connect(self.onDetect)
        self.ui.loadButton.clicked.connect(self.load_model)
        self.ui.detectButton.clicked.connect(self.onDetect)

        self.img_height = 608  # Height of the model input images
        self.img_width = 608  # Width of the model input images
        self.img_channels = 3  # Number of color channels of the model input images
        self.mean_color = [123, 117, 104]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
        self.swap_channels = [2, 1, 0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
        self.n_classes = 1  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
        self.scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
        self.aspect_ratios = [[1.0, 2.0, 0.5],
                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                             [1.0, 2.0, 0.5],
                             [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
        self.two_boxes_for_ar1 = True
        self.steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
        self.offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
        self.clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
        self.variances = [0.1, 0.1, 0.2, 0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
        self.normalize_coords = True

        self.image_path = None
        self.model_path = None

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Select image", "./datasets/image/",
                                                         "All Files (*.*)", options=options)
        self.image = QImage(self.image_path)
        self.ui.label.setPixmap(QPixmap.fromImage(self.image))
        self.ui.label.resize(self.image.width(), self.image.height())
        self.resize(self.image.width(), self.image.height())

    def load_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.model_path = QFileDialog.getOpenFileName(self, "Select model", "./weights/", options=options)
        K.clear_session()  # Clear previous models from memory.
        self.model = ssd_300(image_size=(self.img_height, self.img_width, self.img_channels),
                             n_classes=self.n_classes,
                             mode='training',
                             l2_regularization=0.0005,
                             scales=self.scales,
                             aspect_ratios_per_layer=self.aspect_ratios,
                             two_boxes_for_ar1=self.two_boxes_for_ar1,
                             steps=self.steps,
                             offsets=self.offsets,
                             clip_boxes=self.clip_boxes,
                             variances=self.variances,
                             normalize_coords=self.normalize_coords,
                             subtract_mean=self.mean_color,
                             swap_channels=self.swap_channels)

        # 2: Load some weights into the model.
        # TODO: Set the path to the weights you want to load.
        self.model.load_weights(self.model_path[0], by_name=True)
        # 3: Instantiate an optimizer and the SSD loss function and compile the model.
        #    If you want to follow the original Caffe implementation, use the preset SGD
        #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.
        sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

        QMessageBox.information(self, "Message", "Model loaded!", QMessageBox.Yes)

    def onDetect(self):
        if not self.image_path:
            QMessageBox.information(self, "error", "Image is not selected!", QMessageBox.Yes)
            return
        if not self.model_path:
            QMessageBox.information(self, "error", "Model is not loaded!", QMessageBox.Yes)
            return
        img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_height, self.img_width))
        input_tensor = img[np.newaxis, :, :, :]

        y_pred = self.model.predict(input_tensor)

        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.3,
                                           iou_threshold=0.4,
                                           top_k=200,
                                           normalize_coords=self.normalize_coords,
                                           img_height=self.img_height,
                                           img_width=self.img_width)

        # confidence_threshold = 0.5
        # y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

        # Display the image and draw the predicted boxes onto it.

        # Set the colors for the bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()
        classes = ['background',
                   'ship']

        plt.figure()
        plt.imshow(img)

        current_axis = plt.gca()

        for box in y_pred_decoded[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[2] * img.shape[1] / self.img_width
            ymin = box[3] * img.shape[0] / self.img_height
            xmax = box[4] * img.shape[1] / self.img_width
            ymax = box[5] * img.shape[0] / self.img_height
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
            plt.axis('off')
            plt.savefig('result.png')

        # probability_map, feature_map = self.session.run([self.model.probability_map, self.model.feature_map],
        #                                                 feed_dict={self.inputs: input_tensor})
        # probability_map = np.reshape(probability_map, [probability_map.shape[1], probability_map.shape[2]])
        # feature_map = np.reshape(feature_map, [feature_map.shape[1], feature_map.shape[2], 2])
        # feature_map_pos = feature_map[:, :, 1]
        #
        # suspect_region = np.where(probability_map > PROBABILITY_THRESHOLD)
        # coordinates = np.vstack((suspect_region[1], suspect_region[0])).T  # exchange the x and y coordinates
        # scores = [feature_map_pos[y, x] for x, y in coordinates]
        # coordinates = 8 * coordinates  # mapping to corresponding coordinate in origin image
        #
        # suppressed_coordinate = utils.non_max_suppress(coordinates, scores, WINDOW_SIZE, 0.0)
        #
        # detect_out = img.copy()
        # for coordinate in suppressed_coordinate:
        #     tl = tuple(coordinate)
        #     br = tuple(coordinate + WINDOW_SIZE)
        #     cv2.rectangle(detect_out, tl, br, (255, 255, 255))
        #cv2.imwrite('result.bmp', detect_out)

        self.ui.label.setPixmap(QPixmap.fromImage(QImage('result.png')))
        # self.ui.label.resize(self.image.width(), self.image.height())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())