import tensorflow as tf
from model_utils import FullyConnected,positional_encoding

class EncoderLayer(tf.keras.layers.Layer) :
    def __init__(self,embedding_dim,num_heads,fully_connected_dim,
                 dropout_rate=0.1,layernorm_eps=1e-6) :
        super(EncoderLayer,self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embedding_dim,
            dropout = dropout_rate
        )

        self.ffn = FullyConnected(embedding_dim=embedding_dim,fully_connected_dim=fully_connected_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self,x,training,mask) :
        mha_output = self.mha(x,x,x,mask)
        normal1 = self.layernorm1(x + mha_output)

        ffn_output = self.ffn(normal1)
        encoder_layer_output = self.layernorm2(ffn_output + normal1)
        return encoder_layer_output

class Encoder(tf.keras.layers.Layer) :
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6) :
        super(Encoder,self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size,self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding,self.embedding_dim)

        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps)
                            for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self,x,training,mask) :
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim,tf.float32))
        x += self.pos_encoding[:,:seq_len,:]
        x = self.dropout(x,training=training)
        for i in range(self.num_layers) :
            x = self.enc_layers[i](x,training,mask)
        return x