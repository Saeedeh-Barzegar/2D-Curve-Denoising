
import tensorflow as tf
from tensorflow.keras import layers


def inception_module(x, filters=64):

    branch1 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)

    branch3 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    branch3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(branch3)

    branch5 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    branch5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(branch5)

    branch_pool = layers.MaxPool1D(pool_size=3, strides=1, padding='same')(x)
    branch_pool = layers.Conv1D(filters, 1, padding='same', activation='relu')(branch_pool)

    return layers.concatenate([branch1, branch3, branch5, branch_pool])


def inception_module_transpose(x, filters=64):

    branch1 = layers.Conv1DTranspose(filters, 1, padding='same', activation='relu')(x)

    branch3 = layers.Conv1DTranspose(filters, 1, padding='same', activation='relu')(x)
    branch3 = layers.Conv1DTranspose(filters, 3, padding='same', activation='relu')(branch3)

    branch5 = layers.Conv1DTranspose(filters, 1, padding='same', activation='relu')(x)
    branch5 = layers.Conv1DTranspose(filters, 5, padding='same', activation='relu')(branch5)

    branch_pool = layers.MaxPool1D(pool_size=3, strides=1, padding='same')(x)
    branch_pool = layers.Conv1DTranspose(filters, 1, padding='same', activation='relu')(branch_pool)

    return layers.concatenate([branch1, branch3, branch5, branch_pool])


def dynamic_crop(inputs):

    input_tensor, decoded_tensor = inputs
    return tf.slice(decoded_tensor, [0, 0, 0], [-1, tf.shape(input_tensor)[1], -1])


def build_model():
    """
    The model accepts variable-length input sequences of shape (None, 2)
    and returns denoised sequences of the same length.
    """
    inputs = tf.keras.Input(shape=(None, 2))

    # --- Encoder ---
    x = layers.Conv1D(128, 3, activation='relu')(inputs)

    x = inception_module(x, filters=64)
    x = layers.Conv1D(256, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    x = inception_module(x, filters=64)
    x = layers.Conv1D(512, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    # --- Decoder ---
    x = layers.Conv1DTranspose(512, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D(2)(x)

    x = inception_module_transpose(x, filters=64)
    x = layers.Conv1DTranspose(256, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D(2)(x)

    x = inception_module_transpose(x, filters=64)
    x = layers.Conv1DTranspose(128, 3, activation='relu')(x)

    x = layers.SimpleRNN(64, activation='relu', return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(2, activation='tanh'))(x)

    outputs = layers.Lambda(dynamic_crop)([inputs, x])

    model = tf.keras.Model(inputs, outputs, name='DenoisingAutoencoder')
    return model
