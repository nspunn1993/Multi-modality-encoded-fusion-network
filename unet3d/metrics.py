from functools import partial

from keras import backend as K
import tensorflow as tf

def comb_loss(y_true, y_pred):
    return 0.5*weighted_dice_coefficient_loss(y_true, y_pred) + 0.5*iou_loss(y_true, y_pred)
    
def iou_loss(y_true, y_pred):
    beta=0.7
    epsilon = 1.e-9
    numerator = tf.reduce_sum(y_true * y_pred, axis=(-3,-2,-1))
    #denominator = y_true * y_pred + beta * (1 - y_true) * tf.math.log(1/(y_pred + epsilon)) + (1 - beta) * y_true * tf.math.log(1/(1 - y_pred + epsilon))
    denominator = (y_pred + y_true) - y_true * y_pred

    return 1 - (numerator) / (tf.reduce_sum(denominator, axis=(-3,-2,-1)))

def tloss(y_true, y_pred):
    beta=0.7
    epsilon = 1.e-9
    numerator = tf.reduce_sum(y_true * y_pred, axis=(-3,-2,-1))
    #denominator = y_true * y_pred + beta * (1 - y_true) * tf.math.log(1/(y_pred + epsilon)) + (1 - beta) * y_true * tf.math.log(1/(1 - y_pred + epsilon))
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=(-3,-2,-1)) + 1)

def reg_loss(y_true, y_pred):
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    #ce = tf.multiply(y_true, -tf.math.log(model_out))
    sv = similarity_value(y_true,y_pred)
    ce = tf.add(1.,tf.multiply(-1.,sv))
    weight = tf.pow(tf.add(epsilon,sv),-2)
    
    fl = tf.multiply(weight, ce)
    
    #reduced_fl = tf.reduce_max(fl, axis=1)
    
    return tf.reduce_mean(fl)

def similarity_value(y_true, y_pred):
    epsilon = 1.e-9
    num = tf.multiply(2., tf.add(tf.reduce_sum(tf.multiply(y_true, y_pred)),epsilon))
    den = tf.pow(tf.add(tf.add(tf.reduce_sum(y_true),tf.reduce_sum(y_pred)),epsilon),-1)
    final = tf.multiply(num,den)
    return tf.reduce_mean(final)

def real_similarity_value(y_true, y_pred):
    return 1 - similarity_value(y_true, y_pred)

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return 1-weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
