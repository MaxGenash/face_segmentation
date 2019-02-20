"""
Utilities performing datasets and saving them in appropriate format
"""


import argparse
import os
from glob import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras_applications import imagenet_utils
from scipy.misc import imresize
from scipy.ndimage import imread
from sklearn.model_selection import train_test_split

from config import *

seed = 1

def get_palette(dataset_name):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    # TODO: парсити з csv файлу
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette


# TODO use it
def color_seg(seg, palette):
    """
    Replace classes with their colors.

    Takes:
        seg: H x W segmentation image of class IDs
    Gives:
        H x W x 3 image of class colors
    """
    return palette[seg.flat].reshape(seg.shape + (3,))


# TODO use it
def vis_seg(img, seg, palette, alpha=0.5):
    """
    Visualize segmentation as an overlay on the image.

    Takes:
        img: H x W x 3 image in [0, 255]
        seg: H x W segmentation image of class IDs
        palette: K x 3 colormap for all classes
        alpha: opacity of the segmentation in [0, 1]
    Gives:
        H x W x 3 image with overlaid segmentation
    """
    vis = np.array(img, dtype=np.float32)
    mask = seg > 0
    vis[mask] *= 1. - alpha
    vis[mask] += alpha * palette[seg[mask].flat]
    vis = vis.astype(np.uint8)
    return vis


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Input array scaled to [-1.,1.]
    """
    return imagenet_utils.preprocess_input(x, mode='tf')


def _create_datagen(images, masks, img_gen, mask_gen):
    img_iter = img_gen.flow(images, seed=seed)
    mask_iter = mask_gen.flow(
        masks,
        # in need only hair, uncomment:
        # np.expand_dims(masks[:, :, :, 0], axis=4),
        seed=seed  # use same seed to apply same augmentation with image
    )

    def datagen():
        while True:
            img = img_iter.next()
            mask = mask_iter.next()
            yield img, mask

    return datagen


def load_data(img_file, mask_file, return_generator=False):
    images = np.load(img_file)
    masks = np.load(mask_file)

    x_train, x_val, y_train, y_val = train_test_split(
        images,
        masks,
        test_size=0.2,
        random_state=seed
    )

    train_gen, validation_gen = None, None

    if return_generator:
        train_img_gen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            shear_range=0.2,
            zoom_range=0.2,
            # vertical_flip=True,  # debug
            horizontal_flip=True,
        )
        train_img_gen.fit(images)
        train_mask_gen = ImageDataGenerator(
            rotation_range=20,
            shear_range=0.2,
            zoom_range=0.2,
            # vertical_flip=True,  # debug
            horizontal_flip=True,
        )
        train_gen = _create_datagen(
            x_train,
            y_train,
            img_gen=train_img_gen,
            mask_gen=train_mask_gen
        )

        validation_img_gen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=True,
        )
        validation_img_gen.fit(images)
        validation_mask_gen = ImageDataGenerator(
            horizontal_flip=True,
        )
        validation_gen = _create_datagen(
            x_val,
            y_val,
            img_gen=validation_img_gen,
            mask_gen=validation_mask_gen
        )

    return x_train, x_val, y_train, y_val, images.shape[1:3], train_gen, validation_gen


def create_data(data_dir, out_dir, img_size, debug=False):
    """
    It expects following directory layout in data_dir:
    images/
      0001.jpg
      0002.jpg
    masks/
      0001.ppm
      0002.ppm
    Mask image has 3 colors R, G and B. R is hair. G is face. B is bg.
    Finally, it will create images.npy and masks.npy in out_dir.
    """
    # TODO зробити шляхи і розширення файлів конфігуруємими
    img_files = sorted(glob(data_dir + '/images/*.jpg'))
    mask_files = sorted(glob(data_dir + '/masks/*.ppm'))
    print('Found {0} images and {1} masks'.format(len(img_files), len(mask_files)))

    print('Buliding lists...')
    images_list = []
    masks_list = []
    i = 0
    for img_path, mask_path in zip(img_files, mask_files):
        img = imread(img_path)
        img = imresize(img, img_size)
        img = img / 255  # reshape colors from 0..255 to 0..1

        mask = imread(mask_path)
        mask = imresize(mask, img_size, interp='nearest')
        mask = mask / 255

        # TODO: додати модифікування(flip, zoom, etc) зображень для збільшення датасету

        if debug:
            import matplotlib.pyplot as plt
            plt.suptitle(img_path + ' | ' + mask_path)
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.show()

        images_list.append(img)
        masks_list.append(mask)
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, len(img_files)))
        i += 1

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    images_out_dir_name = out_dir + '/images-{0}x{1}.npy'.format(img_size[0], img_size[1])
    masks_out_dir_name = out_dir + '/masks-{0}x{1}.npy'.format(img_size[0], img_size[1])

    print('saving to disc files \n{0} and \n{1}'.format(images_out_dir_name, masks_out_dir_name))

    np.save(images_out_dir_name, np.array(images_list))
    np.save(masks_out_dir_name, np.array(masks_list))

    print('create_data finished')


if __name__ == '__main__':
    # Note, that image shape should be HxW
    create_data(DATA_PATH, DATA_PATH, (IMAGE_HEIGHT, IMAGE_WIDTH))
