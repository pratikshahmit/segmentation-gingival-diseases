import os

PATH_ROOT = '/mas/u/arana/datasets/kumbh/kumbh_labelled/kumbh_data'
PATH_GIT = '/mas/u/arana/github_mit/periodontal_classifier'
PATH_LOGS = os.path.join(PATH_GIT, 'LOGS')
PATH_CSV = os.path.join(PATH_GIT, 'Controller', 'kumbh_metadata.csv')
PATH_CSV_TRAIN = os.path.join(PATH_GIT, 'Controller', 'TRAIN_CSV.csv')
PATH_CSV_TEST = os.path.join(PATH_GIT, 'Controller', 'TEST_CSV.csv')
PATH_CSV_VALIDATION_DATASET = os.path.join(PATH_GIT, 'Controller', 'validation_phase_CSV.csv')
PATH_NUMPY_DATA = '/mas/u/arana/datasets/kumbh/kumbh_labelled/kumbh_numpy_data_by_score'
PATH_NUMPY_DATA_CH = '/mas/u/arana/datasets/kumbh/kumbh_labelled/kumbh_numpy_data_by_score_CH'
PATH_PREDICTED_DATA_SAVE = os.path.join('/mas/u/arana/datasets/kumbh/kumbh_labelled/predicted_images_validation')
if not os.path.exists(PATH_PREDICTED_DATA_SAVE): os.mkdir(PATH_PREDICTED_DATA_SAVE)
PATH_VALIDATION_DATASET = '/mas/u/arana/datasets/kumbh/kumbh_labelled/perio_frame_validation'
# PATH_PREDICTED_DATA_SAVE_CH = os.path.join('/mas/u/arana/datasets/kumbh/kumbh_labelled/predicted_images_CH')
# if not os.path.exists(PATH_PREDICTED_DATA_SAVE_CH): os.mkdir(PATH_PREDICTED_DATA_SAVE_CH)

PATH_NUMPY_DATA_SCORE_1 = os.path.join(PATH_NUMPY_DATA, '1/im.npy')
PATH_NUMPY_DATA_SCORE_2 = os.path.join(PATH_NUMPY_DATA, '2/im.npy')
PATH_NUMPY_DATA_SCORE_3 = os.path.join(PATH_NUMPY_DATA, '3/im.npy')
PATH_NUMPY_DATA_SCORE_4 = os.path.join(PATH_NUMPY_DATA, '4/im.npy')

PATH_NUMPY_DATA_SCORE_1_labels = os.path.join(PATH_NUMPY_DATA, '1/labels.npy')
PATH_NUMPY_DATA_SCORE_2_labels = os.path.join(PATH_NUMPY_DATA, '2/labels.npy')
PATH_NUMPY_DATA_SCORE_3_labels = os.path.join(PATH_NUMPY_DATA, '3/labels.npy')
PATH_NUMPY_DATA_SCORE_4_labels = os.path.join(PATH_NUMPY_DATA, '4/labels.npy')

# PATH_NUMPY_DATA_SCORE_1_CH = os.path.join(PATH_NUMPY_DATA_CH, '1/im.npy')
# PATH_NUMPY_DATA_SCORE_2_CH = os.path.join(PATH_NUMPY_DATA_CH, '2/im.npy')
# PATH_NUMPY_DATA_SCORE_3_CH = os.path.join(PATH_NUMPY_DATA_CH, '3/im.npy')
# PATH_NUMPY_DATA_SCORE_4_CH = os.path.join(PATH_NUMPY_DATA_CH, '4/im.npy')

# PATH_NUMPY_DATA_SCORE_1_labels_CH = os.path.join(PATH_NUMPY_DATA_CH, '1/labels.npy')
# PATH_NUMPY_DATA_SCORE_2_labels_CH = os.path.join(PATH_NUMPY_DATA_CH, '2/labels.npy')
# PATH_NUMPY_DATA_SCORE_3_labels_CH = os.path.join(PATH_NUMPY_DATA_CH, '3/labels.npy')
# PATH_NUMPY_DATA_SCORE_4_labels_CH = os.path.join(PATH_NUMPY_DATA_CH, '4/labels.npy')

TRAIN_BATCH_SIZE = 32
IMAGE_SIZE = (640, 480)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
N_CHANNELS_IN = 3
N_CHANNELS_OUT = 1
TRAIN_VAL_SPLIT = 1.0

NUM_NN_IMAGE_OUTPUTS = 1
N_CLASSES = 5
FILTERS = [16, 32, 64, 128, 256]


LEARNING_RATE = 1e-6
N_ITERATIONS = 5000

