import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
def tversky_loss(alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        tp = tf.reduce_sum(y_true_f * y_pred_f, axis=[1, 2, 3])
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=[1, 2, 3])
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=[1, 2, 3])
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1.0 - tf.reduce_mean(tversky)
    return loss


@register_keras_serializable()
def dice_loss(smooth: float = 1e-6):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true_f + y_pred_f, axis=[1, 2, 3])
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - tf.reduce_mean(dice)
    return loss


@register_keras_serializable()
def combined_loss(alpha: float = 0.5, beta: float = 0.5):
    tversky = tversky_loss()
    bce = tf.keras.losses.BinaryCrossentropy()

    def loss(y_true, y_pred):
        return alpha * tversky(y_true, y_pred) + beta * bce(y_true, y_pred)
    return loss


def build_loss(loss_config: dict):
    """Seleciona a função de perda por loss_config['name']: tversky, dice ou bce.
    alpha/beta se aplicam só a tversky (peso de FP/FN); combined_loss usa um par
    alpha/beta com significado diferente (peso tversky-vs-bce) e por isso fica fora
    deste dispatch — importe-a diretamente se for usá-la."""
    name = loss_config.get('name', 'tversky')
    if name == 'tversky':
        return tversky_loss(alpha=loss_config.get('alpha', 0.7), beta=loss_config.get('beta', 0.3))
    if name == 'dice':
        return dice_loss()
    if name in ('bce', 'binary_crossentropy'):
        return tf.keras.losses.BinaryCrossentropy()
    raise ValueError(f"training.loss.name desconhecido: {name!r}")


def build_loss_custom_objects(loss_config: dict) -> dict:
    """custom_objects para tf.keras.models.load_model: registra as losses
    serializáveis por nome, mais 'loss' apontando para a configurada atualmente
    (Keras serializa a loss de compile() sob esse nome genérico)."""
    return {
        'tversky_loss': tversky_loss(alpha=loss_config.get('alpha', 0.7), beta=loss_config.get('beta', 0.3)),
        'dice_loss': dice_loss(),
        'loss': build_loss(loss_config),
    }
