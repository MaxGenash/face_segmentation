from __future__ import print_function
import os
import time
from keras.models import load_model
from keras.utils import CustomObjectScope
from scipy.misc import imresize
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import numpy as np
from config import *

if CHECK_MODEL_ON_CPU_ONLY:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import numpy as np
# from PIL import Image
# from data_processing import make_palette, color_seg, vis_seg
#
#
# # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
# im = Image.open('demo/image.jpg')
# in_ = np.array(im, dtype=np.float32)
# in_ = in_[:,:,::-1]
# in_ -= np.array((104.00698793,116.66876762,122.67891434))
# in_ = in_.transpose((2,0,1))
#
# # load net
# net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
# # shape for input (data blob is N x C x H x W), set data
# net.blobs['data'].reshape(1, *in_.shape)
# net.blobs['data'].data[...] = in_
# # run net and take argmax for prediction
# net.forward()
# out = net.blobs['score'].data[0].argmax(axis=0)
#
# # visualize segmentation in PASCAL VOC colors
# voc_palette = make_palette(21)
# out_im = Image.fromarray(color_seg(out, voc_palette))
# out_im.save('demo/output.png')
# masked_im = Image.fromarray(vis_seg(im, out, voc_palette))
# masked_im.save('demo/visualization.jpg')
#
# # TODO додати утиліти для розрахунку meanIOU й pixel acc


def standardize(image, mean=None, std=None):
    if mean is None:
        # These values are available from all images.
        mean = [[29.24429131, 29.24429131, 29.24429131]]
    if std is None:
        # These values are available from all images.
        std = [[69.8833313, 63.37436676, 61.38568878]]
    x = (image - np.array(mean)) / (np.array(std) + 1e-7)
    return x


def evaluate_model(saved_model_to_evaluate_path, image_to_evaluate_path, num_classes, custom_objects):
    if custom_objects:
        with CustomObjectScope(custom_objects(num_classes)):
            model = load_model(saved_model_to_evaluate_path)
    else:
        model = load_model(saved_model_to_evaluate_path)

    img = imread(image_to_evaluate_path)
    img = imresize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image_for_model = img/255
    image_for_model = image_for_model.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS).astype(float)
    # image_for_model = standardize(image_for_model)

    timestamp = time.time()
    prediction = model.predict(image_for_model)
    # prediction = np.rint(prediction * 255).astype(int)
    prediction = prediction.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS)
    # Prediction has each pixel in color which corresponds some class(like green = background, red = hair and so on)
    elapsed = time.time() - timestamp
    print('elapsed1: ', elapsed)

    # TODO: calculate and show metrics
    # dice = np_dice_coef(mask.astype(float) / 255, prediction)
    # print('dice1: ', dice)

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(prediction)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    if MODEL_NAME_TO_EVALUATE == 'MobileUNet':
        from models.MobileUNet import custom_objects
    elif MODEL_NAME_TO_EVALUATE == 'DeeplabV3plus':
        from models.DeeplabV3plus import custom_objects

    evaluate_model(SAVED_MODEL_TO_EVALUATE_PATH, IMAGE_TO_EVALUATE_PATH, NUM_CLASSES, custom_objects)
