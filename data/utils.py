import sentencepiece as spm
import pandas as pd
import tensorflow as tf
class NMTDataset :
    def __init__(self,en_model,th_model):
        self.sp_en = spm.SentencePieceProcessor(model_file = en_model)
        self.sp_th = spm.SentencePieceProcessor(model_file = th_model)
    def encode_th(self,text) :
        return self.sp_th.encode(text,add_bos=True,add_eos=True)
    def encode_en(self,text) :
        return self.sp_en.encode(text,add_bos=True,add_eos=True)
    def decode_th(self,ids) :
        return self.sp_th.decode(ids)
    def decode_en(self,ids) :
        return self.sp_en.decode(ids)
    def create_dataset(self,csv_path,batch_size=64,max_len=150) :
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['en_text', 'th_text'])
        df['en_text'] = df['en_text'].astype(str).apply(self.encode_en)
        df['th_text'] = df['th_text'].astype(str).apply(self.encode_th)
        dataset = tf.data.Dataset.from_tensor_slices(( 
            tf.ragged.constant(df['en_text'].tolist()),
            tf.ragged.constant(df['th_text'].tolist())
            ))
        dataset = dataset.filter(
            lambda en,th : tf.logical_and(
                tf.shape(en)[0] <= max_len,
                tf.shape(th)[0] <= max_len
            )
        )
        dataset = dataset.shuffle(10000).padded_batch(
            batch_size=batch_size,
            padded_shapes=(max_len,max_len),
            padding_values=(0,0)
        ).prefetch(tf.data.AUTOTUNE)

        return dataset


        