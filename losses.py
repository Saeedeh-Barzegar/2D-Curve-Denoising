import tensorflow as tf

def create_mask(sequence):

    return tf.cast(tf.reduce_any(tf.not_equal(sequence, 0.0), axis=-1), tf.float32)


def last_valid_point(y, mask):

    lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    lengths = tf.maximum(lengths, 1)
    indices = lengths - 1
    batch_indices = tf.range(tf.shape(y)[0])
    gather_indices = tf.stack([batch_indices, indices], axis=1)
    return tf.gather_nd(y, gather_indices)


def calculate_laplacian(curve):

    lap_first   = curve[:, 2:3]   - 2 * curve[:, 1:2]   + curve[:, 0:1]
    lap_central = curve[:, 2:]    - 2 * curve[:, 1:-1]   + curve[:, :-2]
    lap_last    = curve[:, -1:]   - 2 * curve[:, -2:-1]  + curve[:, -3:-2]
    return tf.concat([lap_first, lap_central, lap_last], axis=1)


def calculate_chord_length(curve, mask):
 
    masked = curve * tf.expand_dims(mask, axis=-1)
    diffs = masked[:, 1:, :] - masked[:, :-1, :]
    segment_lengths = tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-6)
    return tf.reduce_sum(segment_lengths, axis=-1)


def denoising_loss(y_true, y_pred):
 
    mask = create_mask(y_true)

    # Reconstruction loss (MSE)
    curve_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Smoothness loss (Laplacian)
    laplacian_loss = tf.reduce_mean(
        tf.abs(calculate_laplacian(y_true) - calculate_laplacian(y_pred))
    )

    # Endpoint losses
    start_loss = tf.reduce_mean(tf.square(y_true[:, 0, :] - y_pred[:, 0, :]))
    end_loss   = tf.reduce_mean(tf.square(
        last_valid_point(y_true, mask) - last_valid_point(y_pred, mask)
    ))

    # Chord length loss
    chord_loss = tf.reduce_mean(tf.abs(
        calculate_chord_length(y_true, mask) - calculate_chord_length(y_pred, mask)
    ))

    return curve_loss + laplacian_loss + start_loss + end_loss + 0.01 * chord_loss


def metric_mse(y_true, y_pred):

    return tf.reduce_mean(tf.square(y_true - y_pred))


def metric_laplacian(y_true, y_pred):
    
    return tf.reduce_mean(
        tf.abs(calculate_laplacian(y_true) - calculate_laplacian(y_pred))
    )


def metric_chord_length(y_true, y_pred):
   
    mask_true = create_mask(y_true)
    mask_pred = create_mask(y_pred)
    return tf.reduce_mean(tf.abs(
        calculate_chord_length(y_true, mask_true) -
        calculate_chord_length(y_pred, mask_pred)
    ))


def metric_start_point(y_true, y_pred):
   
    return tf.reduce_mean(tf.square(y_true[:, 0, :] - y_pred[:, 0, :]))


def metric_end_point(y_true, y_pred):
    
    mask = create_mask(y_true)
    return tf.reduce_mean(tf.square(
        last_valid_point(y_true, mask) - last_valid_point(y_pred, mask)
    ))
