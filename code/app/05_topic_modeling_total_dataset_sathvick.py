# -*- coding: utf-8 -*-
"""05_topic_modeling_total_dataset_sathvick.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Orm_wcg-T3kMRf0kVyKICBHEkaXsf_Ur
"""

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import gensim
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

df = pd.read_csv('/content/training1600000.csv',encoding = "ISO-8859-1",header=None)

df.columns = ["polarity",'id','date','query','user','tweet']
df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+',' ',text)
    text = re.sub('[^a-zA-Z]',' ',text)
    text = word_tokenize(text)
    text = [item for item in text if item not in stop_words]
    text = [lemma.lemmatize(w) for w in text]
    text = [i for i in text if len(i)>2]
    text = ' '.join(text)
    return text

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

df_new = df.sample(n=len(df))

df_new['polarity'].value_counts()

df_new['clean_tweet'] = df_new['tweet'].apply(clean_text)

x_train = df_new['clean_tweet']

tweets = df_new['clean_tweet'].tolist()
words = []
for i in range(len(tweets)):
  words.append(tweets[i].split())

import gensim.corpora as corpora
id2word = corpora.Dictionary(words)
texts = words
corpus = [id2word.doc2bow(text) for text in texts]

from pprint import pprint
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10)
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

type(doc_lda)

len(doc_lda)

print(lda_model.print_topics()[0])

