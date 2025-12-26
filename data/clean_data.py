import pandas as pd
from pythainlp.tokenize import word_tokenize
import re
def clean_text(text):
    text = re.sub(r'[^\w\s\u0E00-\u0E7F]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def th_segment(th_txt) :
    tokens = word_tokenize(th_txt,engine='newmm')
    return " ".join(tokens)

df = pd.read_csv('data.csv')
df['en_text'] = df['en_text'].apply(clean_text)
df['th_text'] = df['th_text'].apply(clean_text).apply(th_segment)

df['en_text'].to_csv('en_corpus.txt',index=False,header=False)
df['th_text'].to_csv('th_corpus.txt',index=False,header=False)

df.to_csv('data.csv',index=False)

