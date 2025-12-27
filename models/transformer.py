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
        ffn_output = self.dropout_ffn(ffn_output,training=training)
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
    
class DecoderLayer(tf.keras.layers.Layer) :
    def __init__(self,embedding_dim,num_heads,fully_connected_dim,
                 dropout_rate=0.1,layernorm_eps=1e-6) :
        super(DecoderLayer, self).__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embedding_dim,
            dropout = dropout_rate
        )

        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embedding_dim,
            dropout = dropout_rate
        )

        self.ffn = FullyConnected(embedding_dim=embedding_dim,fully_connected_dim=fully_connected_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)
    def call(self,x,enc_output,training,look_ahead_mask,padding_mask) :

        masked_mha_output = self.mha1(x,x,x,look_ahead_mask)
        Q1 = self.layernorm1(masked_mha_output + x)

        mha_output2 = self.mha2(Q1,enc_output,enc_output,padding_mask)
        normal2 = self.layernorm2(mha_output2 + Q1)

        ffn_output = self.ffn(normal2)
        ffn_output = self.dropout_ffn(ffn_output,training=training)

        decoder_layer_output = self.layernorm3(ffn_output + normal2)
        return decoder_layer_output

class Decoder(tf.keras.layers.Layer) :
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6) :
        super(Decoder,self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size,self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding,self.embedding_dim)

        self.dec_layers = [DecoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps)
                            for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self,x,enc_output,training,look_ahead_mask,padding_mask) :
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim,tf.float32))
        x += self.pos_encoding[:,:seq_len,:]
        x = self.dropout(x,training=training)
        for i in range(self.num_layers) :
            x = self.dec_layers[i](x,enc_output,training,look_ahead_mask,padding_mask)
        return x
    

class Transformer(tf.keras.Model) :

    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, 
            target_vocab_size, max_positional_encoding_input,
            max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers=num_layers,
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            fully_connected_dim=fully_connected_dim,
                            input_vocab_size=input_vocab_size,
                            maximum_position_encoding=max_positional_encoding_input,
                            dropout_rate=dropout_rate,
                            layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers, 
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            fully_connected_dim=fully_connected_dim,
                            target_vocab_size=target_vocab_size, 
                            maximum_position_encoding=max_positional_encoding_target,
                            dropout_rate=dropout_rate,
                            layernorm_eps=layernorm_eps)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size,activation='softmax')

    def call(self,input_sentence,output_sentence,training,enc_padding_mask,look_ahead_mask,dec_padding_mask) :
        enc_output = self.encoder(input_sentence,training,enc_padding_mask)
        dec_output  = self.decoder(output_sentence,enc_output,training,look_ahead_mask,dec_padding_mask)

        final_output = self.final_layer(dec_output)
        return final_output
        