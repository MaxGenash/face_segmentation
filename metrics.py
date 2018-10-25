import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

smooth = 1e-5


def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_f, 0, 1)))

    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))

    return true_positives / (possible_positives + K.epsilon())


def f1_score(y_true, y_pred):
    return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def np_dice_coef(y_true, y_pred):
    tr = y_true.flatten()
    pr = y_pred.flatten()

    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)


# def dice_coef_loss(y_true, y_pred):
#     return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


def greet_curried(greeting):
    def greet(name):
        print(greeting + ', ' + name)
    return greet


# def get_mean_iou_metric(num_classes):
#     def mean_iou(y_true, y_pred):
#         score, up_opt = tf.metrics.mean_iou(y_true, y_pred, num_classes)
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         return score
#     return mean_iou


# def iou_metric(y_true_in, y_pred_in, print_table=False):
#     labels = label(y_true_in > 0.5)
#     y_pred = label(y_pred_in > 0.5)
#
#     true_objects = len(np.unique(labels))
#     pred_objects = len(np.unique(y_pred))
#
#     intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
#
#     # Compute areas (needed for finding the union between all objects)
#     area_true = np.histogram(labels, bins=true_objects)[0]
#     area_pred = np.histogram(y_pred, bins=pred_objects)[0]
#     area_true = np.expand_dims(area_true, -1)
#     area_pred = np.expand_dims(area_pred, 0)
#
#     # Compute union
#     union = area_true + area_pred - intersection
#
#     # Exclude background from the analysis
#     intersection = intersection[1:, 1:]
#     union = union[1:, 1:]
#     union[union == 0] = 1e-9
#
#     # Compute the intersection over union
#     iou = intersection / union
#
#     # Precision helper function
#     def precision_at(threshold, iou):
#         matches = iou > threshold
#         true_positives = np.sum(matches, axis=1) == 1  # Correct objects
#         false_positives = np.sum(matches, axis=0) == 0  # Missed objects
#         false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
#         tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
#         return tp, fp, fn
#
#     # Loop over IoU thresholds
#     prec = []
#     if print_table:
#         print("Thresh\tTP\tFP\tFN\tPrec.")
#     for t in np.arange(0.5, 1.0, 0.05):
#         tp, fp, fn = precision_at(t, iou)
#         if (tp + fp + fn) > 0:
#             p = tp / (tp + fp + fn)
#         else:
#             p = 0
#         if print_table:
#             print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
#         prec.append(p)
#
#     if print_table:
#         print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
#     return np.mean(prec)
# def iou_metric_batch(y_true_in, y_pred_in):
#     batch_size = y_true_in.shape[0]
#     metric = []
#     for batch in range(batch_size):
#         value = iou_metric(y_true_in[batch], y_pred_in[batch])
#         metric.append(value)
#     return np.array(np.mean(metric), dtype=np.float32)
# def my_iou_metric(label, pred):
#     metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
#     return metric_value


class MeanIoU(object):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def mean_iou(self, y_true, y_pred):
        # Wraps np_mean_iou method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self.np_mean_iou, [y_true, y_pred], tf.float32)

    def np_mean_iou(self, y_true, y_pred):
        # Compute the confusion matrix to get the number of true positives,
        # false positives, and false negatives
        # Convert predictions and target from categorical to integer format
        target = np.argmax(y_true, axis=-1).ravel()
        predicted = np.argmax(y_pred, axis=-1).ravel()

        # Trick from torchnet for bincounting 2 arrays together
        # https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        # Compute the IoU and mean IoU from the confusion matrix
        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error and set the value to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 0

        return np.mean(iou).astype(np.float32)

