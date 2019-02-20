TRAIN_ON_CPU_ONLY = False        # In case of GPU has not enough memory
CHECK_MODEL_ON_CPU_ONLY = True   # In case of GPU has not enough memory

# IMAGE_HEIGHT = 224
# IMAGE_WIDTH = 224
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
COLOR_CHANNELS = 3  # 3 = RGB; 1 = grayscale
NUM_CLASSES = 3     # TODO load from {{dataset}}_palette.csv
# DATASET_NAME = 'FASSEG'
DATASET_NAME = 'LFW'

TRAINING_BATCH_SIZE = 128
TRAINING_NUM_EPOCHS = 500
USE_GENERATOR_FOR_TRAINING = True

DATA_PATH = './datasets/lfw_part_labels/'
MODELS_PATH = './assets/trained_models/'
TENSORBOARD_PATH = './assets/tensorboard/'
LOGS_PATH = './assets/logs/'

IMG_FILE_TO_TRAIN_ON = './datasets/lfw_part_labels/images-224x224.npy'
MASK_FILE_TO_TRAIN_ON = './datasets/lfw_part_labels/masks-224x224.npy'

# MobileUNet, DeeplabV3plus, PSPNet50
MODEL_TYPE_TO_TRAIN_ON = 'MobileUNet'
MODEL_TYPE_TO_EVALUATE = 'MobileUNet'

EVALUATION_RESULTS_TO_SAVE_PATH = './assets/evaluation_results/'
# SAVED_MODEL_TO_EVALUATE_PATH = './assets/trained_models/model=DeeplabV3plus__start_date=2018-10-28T19.45.12__batch_size=128__num_epoch=500__steps_per_epoch=18__optimizer=SGD,lr=0.1..0001,decay=4e-5,momentum=0.9,nesterov=True__loss=categorical_crossentropy__with_generator=True__epoch=409__val_loss=0.15.h5'
SAVED_MODEL_TO_EVALUATE_PATH = './assets/trained_models/model=MobileUNet__start_date=2018-10-28T22.32.44__batch_size=16__num_epoch=200__steps_per_epoch=146__optimizer=Adadelta__loss=categorical_crossentropy__with_generator=True__epoch=99__val_loss=0.11.h5'
# SAVED_MODEL_TO_EVALUATE_PATH = './assets/trained_models/model=PSPNet50__start_date=2018-10-25T21.07.09__batch_size=512__num_epoch=100__steps_per_epoch=4__optimizer=Adam,lr=0.001__loss=categorical_crossentropy__with_generator=True__epoch=95__val_loss=0.21.h5'

IMAGE_TO_EVALUATE_PATH = './datasets/lfw_part_labels/images/'
IMAGE_TO_EVALUATE_NAMES = [
    'Aaron_Peirsol_0001.jpg',
    'Manfred_Reyes_Villa_0001.jpg',
    'Abdoulaye_Wade_0004.jpg',
    'Adam_Scott_0001.jpg',
    'Adam_Scott_0002.jpg',
    'Akbar_Hashemi_Rafsanjani_0001.jpg',
    'Billy_Bob_Thornton_0001.jpg',
    'Bill_Callahan_0001.jpg',
    'Bill_Parcells_0002.jpg',
    'Bill_Graham_0007.jpg',
    'Carla_Sullivan_0001.jpg',
    'Ali_Khamenei_0001.jpg',
    'Ali_Hammoud_0001.jpg',
    'Christina_Aguilera_0004.jpg',
    'Barry_Zito_0001.jpg',
    'Baburam_Bhattari_0001.jpg',
    'Ben_Glisan_0001.jpg',
    'Barry_Bonds_0001.jpg',
    'Barrett_Jackman_0002.jpg',
    'AJ_Cook_0001.jpg',
    'Amy_Smart_0001.jpg'
]
