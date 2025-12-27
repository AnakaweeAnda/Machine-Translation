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

def attention(q,k,v,mask = None) :
    matmul_qk = tf.matmul(q,k,transpose_b=True)
    dk = tf.cast(k.shape[1],tf.float32)
    matmul_qk = matmul_qk/tf.math.sqrt(dk)

    if mask :
        matmul_qk += (1-mask) * (-1e9)
    
    attention_weights = tf.keras.activations.softmax(matmul_qk,axis=-1)
    output = tf.matmul(attention_weights,v)

    return output,attention_weights

def FullyConnected(embedding_dim,fully_connected_dim) :
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim,activation='relu'),
        tf.keras.layers.Dense(embedding_dim)
    ])