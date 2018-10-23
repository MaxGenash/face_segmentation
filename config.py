# IMAGE_HEIGHT = 224
# IMAGE_WIDTH = 224
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
COLOR_CHANNELS = 3  # 3 = RGB; 1 = grayscale
NUM_CLASSES = 3     # TODO load from {{dataset}}_palette.csv
# DATASET_NAME = 'FASSEG'
DATASET_NAME = 'LFW'

DATA_PATH = './datasets/lfw_part_labels'
MODELS_PATH = './assets/trained_models'
TENSORBOARD_PATH = './assets/tensorboard'
LOGS_PATH = './assets/logs'

IMG_FILE_TO_TRAIN_ON = './datasets/lfw_part_labels/images-64x64.npy'
MASK_FILE_TO_TRAIN_ON = './datasets/lfw_part_labels/masks-64x64.npy'

SAVED_MODEL_TO_EVALUATE_PATH = './assets/trained_models/MobileUNet_checkpoint_ts=1539671641_epoch=64_val_loss=-0.94.h5'
# IMAGE_TO_EVALUATE_PATH = './datasets/lfw_part_labels/images/Aaron_Peirsol_0001.jpg'
# IMAGE_TO_EVALUATE_PATH = './datasets/FASSEG/V2/Train_RGB/2.bmp'
IMAGE_TO_EVALUATE_PATH = './datasets/IMG_20180921_100129.jpg'
# IMAGE_TO_EVALUATE_PATH = './datasets/IMG_20181002_195342.jpg'
