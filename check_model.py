from __future__ import print_function
import os
import time
from keras.models import load_model
from keras.utils import CustomObjectScope
from scipy.misc import imresize
from scipy.ndimage import imread
import matplotlib.pyplot as plt
from config import *

if CHECK_MODEL_ON_CPU_ONLY:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# TODO standartize LFW datased
def standardize(image, mean=None, std=None):
    if mean is None:
        # These values are available from all images.
        mean = [[29.24429131, 29.24429131, 29.24429131]]
    if std is None:
        # These values are available from all images.
        std = [[69.8833313, 63.37436676, 61.38568878]]
    x = (image - np.array(mean)) / (np.array(std) + 1e-7)
    return x

def evaluate_model(saved_model_to_evaluate_path,
                   image_to_evaluate_path,
                   image_to_evaluate_names,
                   evaluation_results_to_save_path,
                   num_classes,
                   custom_objects):
    if custom_objects:
        with CustomObjectScope(custom_objects(num_classes)):
            model = load_model(saved_model_to_evaluate_path)
    else:
        model = load_model(saved_model_to_evaluate_path)

    model.summary()

    for image_to_evaluate_name in image_to_evaluate_names:
        img = imread(image_to_evaluate_path + image_to_evaluate_name)
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
        print('Image: ', image_to_evaluate_name)
        print('Prediction took: ', elapsed, 'ms')

        # TODO: calculate and show metrics
        # dice = np_dice_coef(mask.astype(float) / 255, prediction)
        # print('dice1: ', dice)
        plt.subplots_adjust(
            top=1,
            bottom=0,
            left=0,
            right=1,
            hspace=0.05,
            wspace=0.05
        )
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.subplots_adjust(
            top=1,
            bottom=0,
            left=0,
            right=1,
            hspace=0.05,
            wspace=0.05
        )
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.axis('off')
        # plt.autoscale(tight=True)
        plt.subplots_adjust(
            top=1,
            bottom=0,
            left=0,
            right=1,
            hspace=0.05,
            wspace=0.05
        )
        plt.imshow(prediction)
        plt.subplots_adjust(
            top=1,
            bottom=0,
            left=0,
            right=1,
            hspace=0.05,
            wspace=0.05
        )
        plt.savefig(
            evaluation_results_to_save_path + image_to_evaluate_name,
            bbox_inches='tight',
            quality=100
        )
        plt.show()


if __name__ == '__main__':
    if MODEL_TYPE_TO_EVALUATE == 'MobileUNet':
        from models.MobileUNet import custom_objects
    elif MODEL_TYPE_TO_EVALUATE == 'DeeplabV3plus':
        from models.DeeplabV3plus import custom_objects
    elif MODEL_TYPE_TO_EVALUATE == 'PSPNet50':
        from models.PSPNet50 import custom_objects

    evaluate_model(
        SAVED_MODEL_TO_EVALUATE_PATH,
        IMAGE_TO_EVALUATE_PATH,
        IMAGE_TO_EVALUATE_NAMES,
        EVALUATION_RESULTS_TO_SAVE_PATH,
        NUM_CLASSES,
        custom_objects
    )
