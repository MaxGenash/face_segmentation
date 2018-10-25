import os
import datetime
import h5py
import math
import pickle
import numpy as np
import pandas as pd
import cv2
import threading
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import misc, ndimage
from sklearn import model_selection, preprocessing, metrics
from sklearn.utils import shuffle
from skimage import transform
from tqdm import tqdm
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger, LearningRateScheduler
from keras import backend as K, optimizers
from keras.losses import binary_crossentropy
import keras.backend.tensorflow_backend as KTF
# import tensorflow as tf
from tensorflow.python.client import device_lib
from config import *
from data_processing import load_data
from metrics import dice_coef, recall, precision, dice_coef_loss, MeanIoU
from models.DeeplabV3plus import DeeplabV3plus
from models.MobileUNet import MobileUNet
from models.unet import get_unet_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize

if TRAIN_ON_CPU_ONLY:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# train_masks_df = pd.read_csv(TRAIN_MASKS_CSV_PATH)
# print('train_masks_df.shape', train_masks_df.shape)


#
# new_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)
# mask_shape = (new_shape[0], new_shape[1], 1)
#
#
# def get_img_id(img_path):
#     return img_path[:15]
#
#
# img_ids = list(map(get_img_id, list(train_masks_df.img.values)))
#
#
# def load_image_disk(img_id, folder=TRAIN_PATH):
#     img = misc.imread(os.path.join(folder, img_id + ".bmp"))
#     return img
#
#
# def get_image(img_id):
#     return train_imgs[img_id]
#
#
# # Return mask as 1/0 binary img with single channel
# def load_mask_disk(img_id, folder=TRAIN_MASKS_PATH, filetype='bmp'):
#     mask = misc.imread(os.path.join(folder, "{}_mask.{}".format(img_id, filetype)), flatten=True)
#     mask[mask > 128] = 1
#     if len(mask.shape) == 2:
#         mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
#     return mask
#
#
# def get_mask(img_id):
#     return train_masks[img_id]
#
#
# # Helper functions to plot car, mask, masked_car
# def plot_image(img_id):
#     img = misc.imread(os.path.join(TRAIN_PATH, img_id + ".bmp"))
#     imgplot = plt.imshow(img)
#     plt.axis('off')
#     plt.show()
#
#
# def plot_mask(img_id, folder=TRAIN_MASKS_PATH, filetype='bmp', ax=None):
#     mask = misc.imread(os.path.join(folder, "{}_mask.{}".format(img_id, filetype)))
#     if ax == None:
#         imgplot = plt.imshow(mask)
#         plt.axis('off')
#         plt.show()
#     else:
#         ax.imshow(mask)
#         ax.axis('off')
#
#
# def plot_masked_image(img_id, ax=None):
#     img = misc.imread(os.path.join(TRAIN_PATH, img_id + ".bmp"))
#     mask = misc.imread(os.path.join(TRAIN_MASKS_PATH, img_id + ".bmp"))
#     mask = mask[:, :, 0:3]
#     mask[mask == 255] = 1
#     masked_img = img * mask
#     if ax == None:
#         imgplot = plt.imshow(masked_img)
#         plt.axis('off')
#         plt.show()
#     else:
#         ax.imshow(masked_img)
#         ax.axis('off')
#
#
# # def gray2rgb(img):
# #     img = np.squeeze(img)
# #     w, h = img.shape
# #     ret = np.empty((w, h, 3), dtype=np.uint8)
# #     ret[:, :, 0] = img
# #     ret[:, :, 1] = img
# #     ret[:, :, 2] = img
# #     return ret
#
# def resize_img(img, new_s=new_shape):
#     return transform.resize(img, new_s)
#
#
# train_imgs = {}
# for img_path in tqdm(os.listdir(TRAIN_PATH)):
#     img_id = get_img_id(img_path)
#     train_imgs[img_id] = cv2.resize(load_image_disk(img_id), (new_shape[0], new_shape[1]))
#
# train_masks = {}
# for img_path in tqdm(os.listdir(TRAIN_MASKS_PATH)):
#     img_id = get_img_id(img_path)
#     train_masks[img_id] = np.expand_dims(cv2.resize(load_mask_disk(img_id), (new_shape[0], new_shape[1])), axis=2)


#
# def generate_training_batch(data, batch_size):
#     while True:
#         X_batch = []
#         Y_batch = []
#         batch_ids = np.random.choice(data,
#                                      size=batch_size,
#                                      replace=False)
#         for idx, img_id in enumerate(batch_ids):
#             x = get_image(img_id)
#             y = get_mask(img_id)
#             x, y = randomShiftScaleRotate(x, y,
#                                           shift_limit=(-0.0625, 0.0625),
#                                           scale_limit=(-0.1, 0.1),
#                                           rotate_limit=(-0, 0))
#             #             x = randomHueSaturationValue(x,
#             #                                hue_shift_limit=(-50, 50),
#             #                                sat_shift_limit=(-5, 5),
#             #                                val_shift_limit=(-15, 15))
#             X_batch.append(x)
#             Y_batch.append(y)
#         X = np.asarray(X_batch, dtype=np.float32)
#         Y = np.asarray(Y_batch, dtype=np.float32)
#         yield X, Y
#
#
# def generate_validation_batch(data, batch_size):
#     while True:
#         X_batch = []
#         Y_batch = []
#         batch_ids = np.random.choice(data,
#                                      size=batch_size,
#                                      replace=False)
#         for idx, img_id in enumerate(batch_ids):
#             x = get_image(img_id)
#             y = get_mask(img_id)
#             X_batch.append(x)
#             Y_batch.append(y)
#         X = np.asarray(X_batch, dtype=np.float32)
#         Y = np.asarray(Y_batch, dtype=np.float32)
#         yield X, Y
#

# def generate_validation_data_seq(data):
#     idx = 0
#     while True:
#         img_id = data[idx]
#         X = get_image(img_id)
#         Y = get_mask(img_id)
#         yield img_id, X, Y
#         idx += 1
#         if idx >= len(data):
#             break


def _lr_schedule(epoch, epochs, lr_base, lr_power, mode):
    if mode is 'power_decay':
        lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    if mode is 'exp_decay':
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        if epoch > 0.9 * epochs:
            lr = 0.0001
        elif epoch > 0.75 * epochs:
            lr = 0.001
        elif epoch > 0.5 * epochs:
            lr = 0.01
        else:
            lr = 0.1

    print('lr: %f' % lr)

    return lr


def create_lr_schedule(epochs, lr_base, lr_power=0.9, mode='power_decay'):
    return lambda epoch: _lr_schedule(epoch, epochs, lr_base, lr_power, mode)


def get_model_memory_usage(batch_size, model):
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    total_memory = 4 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = round(total_memory / (1024 ** 3), 3)
    mbytes = round(total_memory / (1024 ** 2), 3)

    print('trainable_count', trainable_count, 'non_trainable_count', non_trainable_count, 'gbytes', gbytes, 'mbytes',
          mbytes)


# generator that we will use to read the data from the directory
# def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels):
#     while True:
#         ix = np.random.choice(np.arange(len(lists)), batch_size)
#         imgs = []
#         labels = []
#         for i in ix:
#             # images
#             original_img = cv2.imread(img_dir + lists.iloc[i, 0]+".jpg")[:, :, ::-1]
#             resized_img = cv2.resize(original_img, dims+[3])
#             array_img = img_to_array(resized_img)/255
#             imgs.append(array_img)
#             # masks
#             original_mask = cv2.imread(mask_dir + lists.iloc[i, 0] + '.png')
#             resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
#             array_mask = binarylab(resized_mask[:, :, 0], dims, n_labels)
#             labels.append(array_mask)
#         imgs = np.array(imgs)
#         labels = np.array(labels)
#         yield imgs, labels


def train(img_file, mask_file, model_name, num_epochs, batch_size, num_classes, use_generator=False):
    """Training new model"""

    x_train, x_val, y_train, y_val, img_shape, train_gen, validation_gen =\
        load_data(img_file, mask_file, use_generator)
    start_date = str(datetime.datetime.now()).replace(' ', 'T').replace(':', '.')[:-7]
    steps_per_epoch = int(len(x_train) / batch_size)
    validation_steps_per_epoch = int(len(x_val) / batch_size)

    lr_base = 0.01  # * (float(batch_size) / 16)  # For UNet
    # lr_base = 0.05  # For DeepLabV3

    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)               # For UNet
    # optimizer = optimizers.SGD(lr=lr_base, decay=4e-5, momentum=0.9, nesterov=True)    # For DeepLabV3
    # optimizer = Adam(lr=0.001)
    # optimizer=optimizers.RMSprop()

    run_name = 'model={}__start_date={}__batch_size={}__num_epoch={}__steps_per_epoch={}__optimizer={}__loss={}__with_generator={}'.format(
        model_name,
        start_date,
        batch_size,
        num_epochs,
        steps_per_epoch,
        # 'Adam,lr=0.001',
        'SGD,lr=0.0001,momentum=0.9,nesterov=True',
        'categorical_crossentropy',
        use_generator
    )
    tensorboard_loc = os.path.join(TENSORBOARD_PATH, run_name)
    csv_logger_loc = os.path.join(LOGS_PATH, run_name)
    checkpoint_loc = os.path.join(MODELS_PATH, run_name + '__epoch={epoch:02d}__val_loss={val_loss:.2f}.h5')

    if model_name == 'MobileUNet':
        model = MobileUNet(
            num_classes=NUM_CLASSES,
            # Note, that image shape should be HxW
            input_shape=(img_shape[0], img_shape[1], 3),
            alpha=1,
            alpha_up=0.25
        )
    elif model_name == 'DeeplabV3plus':
        model = DeeplabV3plus(input_shape=(img_shape[0], img_shape[1], 3), num_classes=NUM_CLASSES, OS=16)
    else:
        raise ValueError(
            'Invalid argument model_name.'
            'Expected one of ("MobileUNet", "DeeplabV3plus"), received "{0}"'.format(model_name)
        )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        # loss=dice_coef_loss,
        metrics=[dice_coef, recall, precision, MeanIoU(num_classes).mean_iou, 'categorical_crossentropy'],
    )
    print(model.summary())
    get_model_memory_usage(batch_size, model)

    # lr_scheduler is useful in case of SGD
    lr_scheduler = LearningRateScheduler(
        create_lr_schedule(num_epochs, lr_base=lr_base, mode='progressive_drops')
    )
    tensorboard = TensorBoard(log_dir=tensorboard_loc, histogram_freq=0, write_graph=True, write_images=True)
    # tensorboard = TensorBoard(
    #     log_dir=tensorboard_loc, histogram_freq=10, write_grads=True, write_graph=True, write_images=True
    # )
    csv_logger = CSVLogger('{0}.csv'.format(csv_logger_loc))
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        filepath=checkpoint_loc,
        mode='min',
        save_best_only=True,
        verbose=1,
    )
    # early_stopping = callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=5,
    #     verbose=1,
    #     min_delta=0.0001,
    #     mode='min',
    # )
    callbacks_list = [lr_scheduler, tensorboard, checkpoint, csv_logger]

    # Note that using generator uses less memory, but needs a bit more CPU.
    # Sometimes it may help to prevent overfitting because data is always different
    if use_generator:
        print('Gonna train model with generator')
        model.fit_generator(
            generator=train_gen(),
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            validation_data=validation_gen(),
            validation_steps=validation_steps_per_epoch,
            callbacks=callbacks_list,
        )
    else:
        print('Gonna train model from lists')
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=callbacks_list,
            validation_data=(x_val, y_val),
            # steps_per_epoch=steps_per_epoch,
            # validation_steps=validation_steps_per_epoch
        )

    # model.save(trained_model_path)


if __name__ == '__main__':
    train(IMG_FILE_TO_TRAIN_ON, MASK_FILE_TO_TRAIN_ON, MODEL_NAME_TO_TRAIN_ON, 100, 32, NUM_CLASSES, True)
