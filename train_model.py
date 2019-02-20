import os
import datetime
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
from metrics import dice_coef, recall, precision, dice_coef_loss, MeanIoU, f1_score
from models.DeeplabV3plus import DeeplabV3plus
from models.MobileUNet import MobileUNet
from models.PSPNet50 import PSPNet50
from models.unet import get_unet_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize

if TRAIN_ON_CPU_ONLY:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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


def train(img_file, mask_file, model_name, num_epochs, batch_size, num_classes, use_generator=False):
    """Training new model"""

    x_train, x_val, y_train, y_val, img_shape, train_gen, validation_gen =\
        load_data(img_file, mask_file, use_generator)
    start_date = str(datetime.datetime.now()).replace(' ', 'T').replace(':', '.')[:-7]
    steps_per_epoch = int(len(x_train) / batch_size)
    validation_steps_per_epoch = int(len(x_val) / batch_size)

    # optimizer = optimizers.SGD(lr=0.0001, decay=4e-5, momentum=0.9, nesterov=True)
    optimizer = optimizers.Adadelta()  # Adam(lr=0.001), optimizers.RMSprop()

    run_name = 'model={}__start_date={}__batch_size={}__num_epoch={}__steps_per_epoch={}__optimizer={}__loss={}__with_generator={}'.format(
        model_name,
        start_date,
        batch_size,
        num_epochs,
        steps_per_epoch,
        # 'Adam,lr=0.001',
        # 'RMSprop',
        'Adadelta',
        # 'SGD,lr=0.1..0001,decay=4e-5,momentum=0.9,nesterov=True',
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
        model = DeeplabV3plus(input_shape=(img_shape[0], img_shape[1], 3), num_classes=NUM_CLASSES, OS=8)
    elif model_name == 'PSPNet50':
        model = PSPNet50(input_shape=(img_shape[0], img_shape[1], 3), num_classes=NUM_CLASSES)
    else:
        raise ValueError(
            'Invalid argument model_name.'
            'Expected one of ("MobileUNet", "DeeplabV3plus", "PSPNet50"), received "{0}"'.format(model_name)
        )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        # loss=dice_coef_loss,
        metrics=[dice_coef, precision, MeanIoU(num_classes).mean_iou, f1_score, 'categorical_crossentropy'],
    )
    print(model.summary())
    get_model_memory_usage(batch_size, model)

    lr_base = 0.001  # * (float(batch_size) / 16)
    lr_scheduler = LearningRateScheduler(
        create_lr_schedule(num_epochs, lr_base=lr_base, mode='progressive_drops')
    )
    tensorboard = TensorBoard(log_dir=tensorboard_loc, histogram_freq=0, write_graph=True, write_images=True)
    csv_logger = CSVLogger('{0}.csv'.format(csv_logger_loc))
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        filepath=checkpoint_loc,
        mode='min',
        save_best_only=True,
        verbose=1,
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        min_delta=0.0001,
        mode='min',
    )
    callbacks_list = [
        # lr_scheduler,
        # early_stopping,
        tensorboard,
        checkpoint,
        csv_logger
    ]

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


if __name__ == '__main__':
    train(
        IMG_FILE_TO_TRAIN_ON,
        MASK_FILE_TO_TRAIN_ON,
        MODEL_TYPE_TO_TRAIN_ON,
        TRAINING_NUM_EPOCHS,
        TRAINING_BATCH_SIZE,
        NUM_CLASSES,
        USE_GENERATOR_FOR_TRAINING
    )
