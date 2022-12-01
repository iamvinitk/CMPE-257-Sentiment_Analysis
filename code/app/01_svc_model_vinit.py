#!/usr/bin/env python
# coding: utf-8

# In[15]:


from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.svm import SVC


# In[16]:


# read csv
train_data = pd.read_csv('./data/reduced_train.csv')
test_data = pd.read_csv('./data/reduced_test.csv')


# In[17]:


# label the sentiments
encoded_labels = preprocessing.LabelEncoder()

train_labels = encoded_labels.fit_transform(train_data['sentiment'])
test_labels = encoded_labels.fit_transform(test_data['sentiment'])


# In[18]:


# TF-IDF
tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                        token_pattern=r'\w{1,}',
                        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')

tfidf.fit(list(train_data['tweet']) + list(test_data['tweet']))

train_features = tfidf.transform(train_data['tweet'])
test_features = tfidf.transform(test_data['tweet'])


# In[19]:


# SVD
svd = TruncatedSVD(n_components=300)
svd.fit(train_features)

train_features = svd.transform(train_features)
test_features = svd.transform(test_features)


# In[20]:


# Normalize data with StandardScaler
scaler = preprocessing.StandardScaler()
scaler.fit(train_features)

train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)


# In[21]:


# SVC model
svc = SVC(C=1.0, probability=True)

svc.fit(train_features, train_labels)

# predict
predictions = svc.predict(test_features)


# In[22]:


# calculate f1 score
f1 = f1_score(test_labels, predictions, average='weighted')
print(f1)


# In[ ]:




