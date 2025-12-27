import numpy as np
import tensorflow as tf
def positional_encoding(positions,d_model) :
    position = np.arange(positions)[:,np.newaxis]
    k = np.arange(d_model)[np.newaxis,:]
    i = k//2
    angle_rates = 1 / np.power(10000,(2*i)/np.float32(d_model))
    angle_rads = position * angle_rates

    angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])
    angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])

    pos_encoding = angle_rads[np.newaxis,...]
    return tf.cast(pos_encoding,dtype=tf.float32)

def padding_mask(decoder_token_ids) :
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids,0),tf.float32)
    return seq[:,tf.newaxis,:]

def look_ahead_mask(sequence_length) :
    mask = tf.linalg.band_part(tf.ones((1,sequence_length,sequence_length)),-1,0)
    return mask