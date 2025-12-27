import tensorflow as tf
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import Transformer
from models.model_utils import padding_mask, look_ahead_mask
from data.utils import NMTDataset

NUM_LAYERS = 4
EMBEDDING_DIM = 128
NUM_HEADS = 8
FULLY_CONNECTED_DIM = 512
DROPOUT_RATE = 0.1
EPOCHS = 20
BATCH_SIZE = 64
MAX_LEN = 150

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
EN_MODEL_PATH = os.path.join(DATA_DIR, 'en_spm.model')
TH_MODEL_PATH = os.path.join(DATA_DIR, 'th_spm.model')
CSV_PATH = os.path.join(DATA_DIR, 'data.csv')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def create_masks(inp, tar):
    enc_padding_mask = padding_mask(inp)
    dec_padding_mask = padding_mask(inp)
    look_ahead = look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = padding_mask(tar)
    combined_mask = tf.minimum(dec_target_padding_mask, look_ahead)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none'
)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.int32))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

@tf.function
def train_step(transformer, optimizer, inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]  
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    
    with tf.GradientTape() as tape:
        predictions = transformer(
            inp, tar_inp, True,
            enc_padding_mask, combined_mask, dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)
    
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

def main():
    nmt_dataset = NMTDataset(EN_MODEL_PATH, TH_MODEL_PATH)
    train_dataset = nmt_dataset.create_dataset(CSV_PATH, batch_size=BATCH_SIZE, max_len=MAX_LEN)
    
    input_vocab_size = nmt_dataset.sp_en.get_piece_size()
    target_vocab_size = nmt_dataset.sp_th.get_piece_size()
    
    print(f"English vocab size: {input_vocab_size}")
    print(f"Thai vocab size: {target_vocab_size}")
    
    transformer = Transformer(
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        fully_connected_dim=FULLY_CONNECTED_DIM,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_positional_encoding_input=MAX_LEN,
        max_positional_encoding_target=MAX_LEN,
        dropout_rate=DROPOUT_RATE
    )
    
    learning_rate = CustomSchedule(EMBEDDING_DIM)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)
    
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"Restored from {checkpoint_manager.latest_checkpoint}")
    
    for epoch in range(EPOCHS):
        train_loss.reset_state()
        train_accuracy.reset_state()
        
        for batch, (inp, tar) in enumerate(train_dataset):
            train_step(transformer, optimizer, inp, tar)
            
            if batch % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch}, "
                      f"Loss: {train_loss.result():.4f}, "
                      f"Accuracy: {train_accuracy.result():.4f}")
        
        ckpt_path = checkpoint_manager.save()
        print(f"Epoch {epoch + 1} completed. "
              f"Loss: {train_loss.result():.4f}, "
              f"Accuracy: {train_accuracy.result():.4f}")
        print(f"Checkpoint saved: {ckpt_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
