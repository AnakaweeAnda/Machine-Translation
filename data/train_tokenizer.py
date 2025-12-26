import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='en_corpus.txt',
    model_prefix='en_spm',
    vocab_size=8000,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    pad_piece='[PAD]', unk_piece='[UNK]',
    bos_piece='[SOS]', eos_piece='[EOS]',
    model_type='bpe'
)

spm.SentencePieceTrainer.train(
    input='th_corpus.txt',
    model_prefix='th_spm',
    vocab_size=16000,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    pad_piece='[PAD]', unk_piece='[UNK]',
    bos_piece='[SOS]', eos_piece='[EOS]',
    model_type='bpe'
)