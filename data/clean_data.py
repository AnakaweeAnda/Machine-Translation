import pandas as pd
from pythainlp.tokenize import word_tokenize

def th_segment(th_txt) :
    tokens = word_tokenize(th_txt,engine='newmm')
    return " ".join(tokens)

df = pd.read_csv('data.csv')
df['en_text'] = df['en_text']
df['th_text'] = df['th_text'].apply(th_segment)

df['en_text'].to_csv('en_corpus.txt',index=False,header=False)
df['th_text'].to_csv('th_corpus.txt',index=False,header=False)



