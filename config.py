TRAIN_ON_CPU_ONLY = False   # In case of GPU has not enough memory
CHECK_MODEL_ON_CPU_ONLY = True   # In case of GPU has not enough memory

# IMAGE_HEIGHT = 224
# IMAGE_WIDTH = 224
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
COLOR_CHANNELS = 3  # 3 = RGB; 1 = grayscale
NUM_CLASSES = 3     # TODO load from {{dataset}}_palette.csv
# DATASET_NAME = 'FASSEG'
DATASET_NAME = 'LFW'

DATA_PATH = './datasets/lfw_part_labels/'
MODELS_PATH = './assets/trained_models/'
TENSORBOARD_PATH = './assets/tensorboard/'
LOGS_PATH = './assets/logs/'

IMG_FILE_TO_TRAIN_ON = './datasets/lfw_part_labels/images-64x64.npy'
MASK_FILE_TO_TRAIN_ON = './datasets/lfw_part_labels/masks-64x64.npy'

# MODEL_NAME_TO_TRAIN_ON = 'MobileUNet'
MODEL_NAME_TO_EVALUATE = 'MobileUNet'
# MODEL_NAME_TO_EVALUATE = 'DeeplabV3plus'
MODEL_NAME_TO_TRAIN_ON = 'DeeplabV3plus'

SAVED_MODEL_TO_EVALUATE_PATH = './assets/trained_models/model=MobileUNet__start_date=2018-10-25T17.00.42__batch_size=16__num_epoch=100__steps_per_epoch=146__optimizer=SGD,lr=0.0001,momentum=0.9,nesterov=True__loss=categorical_crossentropy__with_generator=True__epoch=99__val_loss=0.12.h5'
# IMAGE_TO_EVALUATE_PATH = './datasets/lfw_part_labels/images/Aaron_Peirsol_0001.jpg'
# IMAGE_TO_EVALUATE_PATH = './datasets/FASSEG/V2/Train_RGB/2.bmp'
IMAGE_TO_EVALUATE_PATH = './datasets/IMG_20180921_100129.jpg'
# IMAGE_TO_EVALUATE_PATH = './datasets/IMG_20181002_195342.jpg'
