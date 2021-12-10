import tensorflow as tf


scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(labels, predictions):
    loss = scce(labels, predictions)
    
    mask = tf.math.logical_not(tf.math.equal(labels, 0))
    loss *= tf.cast(mask, dtype=loss.dtype)
    
    return tf.reduce_mean(loss)