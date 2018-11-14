from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras.preprocessing import image
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

img_dir = "/home/jerry/Datasets/sea_ship/new_train_1109/patch2m5_rgb/test/image_ship"
save_dir = "/home/jerry/Projects/ssd_keras/results/ssd7/2.5/gray/"
img_list = os.listdir(img_dir)


# Set a few configuration parameters.
model_mode = 'inference_fast'
img_height = 300 # Height of the input images
img_width = 300 # Width of the input images
img_channels = 1     # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 3 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode=model_mode,
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = 'weights/seaship/new_train_1109/ssd7_seaship2.5_gray_epoch-115_loss-0.8034_val_loss-0.6399.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

num_of_images = len(img_list)
cnt = 0

for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    orig_images = []  # Store the images here.
    input_images = []  # Store resized versions of the images here.
    orig_image = cv2.imread(img_path)
    #uncomment if testing grayscale images
    orig_image = np.mean(orig_image, -1, keepdims=True)
    orig_image = np.concatenate([orig_image, ]*3, -1)
    orig_image = orig_image.astype(np.uint8)

    orig_images.append(orig_image)
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    #uncomment if testing grayscale images
    input_images = np.mean(input_images, -1, keepdims=True)

    #make predictions
    y_pred = model.predict(input_images)

    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    print("processed {}/{} images ".format(str(cnt), str(num_of_images)))
    #np.set_printoptions(precision=2, suppress=True, linewidth=90)
    # print("Predicted boxes:\n")
    # print('   class   conf xmin   ymin   xmax   ymax')
    #print(y_pred_thresh[0])
    #y_pred_write = y_pred_thresh[0][2:] + y_pred_thresh[0][:1]

    with open(os.path.join(save_dir, "results.txt"), 'a+') as f:
        f.write(img_name)
        f.write('\t')
        for pred in y_pred_thresh[0]:
            pred_write = np.concatenate((pred[2:], pred[:2]), axis=-1)
            for x in pred_write:
                f.write(str(x))
                f.write('\t')
        f.write("\n")

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()
    classes = ['background',
               'ship', 'carrier', 'warship']

    plt.figure(figsize=(12, 12))
    plt.imshow(orig_images[0])
    current_axis = plt.gca()
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / img_width
        ymin = box[3] * orig_images[0].shape[0] / img_height
        xmax = box[4] * orig_images[0].shape[1] / img_width
        ymax = box[5] * orig_images[0].shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
    plt.savefig(os.path.join(save_dir, "images", img_name))
    plt.close("all")
    cnt += 1



