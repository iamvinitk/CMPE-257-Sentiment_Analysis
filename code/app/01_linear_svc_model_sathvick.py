#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords


# In[2]:


import pandas as pd
import numpy as np
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


# In[3]:


df = pd.read_csv('/content/training1600000.csv',encoding = "ISO-8859-1",header=None)


# In[4]:


df.columns = ["polarity",'id','date','query','user','tweet']
df


# ## Text Cleaning

# In[5]:


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


# In[6]:


stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()


# In[7]:


# df_new = df[0:100000]
df_new = df.sample(n=100000)


# In[8]:


df_new['polarity'].value_counts()


# In[9]:


df_new['clean_tweet'] = df_new['tweet'].apply(clean_text)


# In[10]:


x_train = df_new['clean_tweet']


# In[11]:


tfidf_vectorizer = TfidfVectorizer(min_df=10)

x_train_tfidf = pd.DataFrame(tfidf_vectorizer.fit_transform(x_train).toarray(), columns=tfidf_vectorizer.get_feature_names_out())

display(x_train_tfidf.shape)
display(x_train_tfidf.head())


# In[12]:


model = LinearSVC()


# In[13]:


y_train = df_new.polarity


# In[14]:


# y_train


# In[15]:


df_test = pd.read_csv('/content/testdata.manual.2009.06.14.csv',header=None)


# In[16]:


df_test.columns = ["polarity",'id','date','query','user','tweet']
df_test


# In[17]:


df_test['clean_tweet'] = df_test['tweet'].apply(clean_text)


# In[18]:


x_test = df_test['clean_tweet']
x_test_tfidf = pd.DataFrame(tfidf_vectorizer.transform(x_test).toarray(), columns=tfidf_vectorizer.get_feature_names_out())


# In[19]:


y_test = df_test['polarity']


# In[20]:


import time
start_time = time.time()

model.fit(x_train_tfidf, y_train)
results = {
    'time_to_train': [],
    'time_to_test': [],
    'accuracy': [],
    'f1': []
}
# training end
end_time = time.time()
time_to_train = end_time - start_time
print('training completed:', '{:.2f}'.format(time_to_train), 'seconds')

# testing start
print('testing...')
start_time = time.time()

# make predictions on validation set
y_pred = model.predict(x_test_tfidf)

# testing end
end_time = time.time()
time_to_test = end_time - start_time
print('testing completed:', '{:.2f}'.format(time_to_test), 'seconds\n')

# add results to result map
results['time_to_train'].append(time_to_train)
results['time_to_test'].append(time_to_test)
results['accuracy'].append(accuracy_score(y_test, y_pred))
results['f1'].append(f1_score(y_test, y_pred, average=None))


# In[21]:


results


# In[23]:


from sklearn.linear_model import SGDClassifier


# In[24]:


model2 = SGDClassifier(max_iter=10000, tol=1e-3)


# In[25]:


import time
start_time = time.time()

model2.fit(x_train_tfidf, y_train)
results_SGD = {
    'time_to_train': [],
    'time_to_test': [],
    'accuracy': [],
    'f1': []
}
# training end
end_time = time.time()
time_to_train = end_time - start_time
print('training completed:', '{:.2f}'.format(time_to_train), 'seconds')

# testing start
print('testing...')
start_time = time.time()

# make predictions on validation set
y_pred = model.predict(x_test_tfidf)

# testing end
end_time = time.time()
time_to_test = end_time - start_time
print('testing completed:', '{:.2f}'.format(time_to_test), 'seconds\n')

# add results to result map
results_SGD['time_to_train'].append(time_to_train)
results_SGD['time_to_test'].append(time_to_test)
results_SGD['accuracy'].append(accuracy_score(y_test, y_pred))
results_SGD['f1'].append(f1_score(y_test, y_pred, average=None))


# In[26]:


results_SGD

