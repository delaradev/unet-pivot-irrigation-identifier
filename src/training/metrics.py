import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
def iou_score(threshold: float = 0.5, smooth: float = 1e-6, name='iou_score'):
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_bin = tf.where(y_pred >= threshold, 1.0, 0.0)
        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true + y_pred_bin,
                              axis=[1, 2, 3]) - intersection
        return tf.reduce_mean((intersection + smooth) / (union + smooth))
    metric.__name__ = name
    return metric


@register_keras_serializable()
def dice_score(threshold: float = 0.5, smooth: float = 1e-6, name='dice_score'):
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_bin = tf.where(y_pred >= threshold, 1.0, 0.0)
        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2, 3])
        total = tf.reduce_sum(y_true + y_pred_bin, axis=[1, 2, 3])
        return tf.reduce_mean((2.0 * intersection + smooth) / (total + smooth))
    metric.__name__ = name
    return metric


@register_keras_serializable()
def precision_score(threshold: float = 0.5, smooth: float = 1e-6, name='precision_score'):
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_bin = tf.where(y_pred >= threshold, 1.0, 0.0)
        tp = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2, 3])
        fp = tf.reduce_sum((1 - y_true) * y_pred_bin, axis=[1, 2, 3])
        return tf.reduce_mean((tp + smooth) / (tp + fp + smooth))
    metric.__name__ = name
    return metric


@register_keras_serializable()
def recall_score(threshold: float = 0.5, smooth: float = 1e-6, name='recall_score'):
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_bin = tf.where(y_pred >= threshold, 1.0, 0.0)
        tp = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2, 3])
        fn = tf.reduce_sum(y_true * (1 - y_pred_bin), axis=[1, 2, 3])
        return tf.reduce_mean((tp + smooth) / (tp + fn + smooth))
    metric.__name__ = name
    return metric
