# -*- coding: utf-8 -*-
"""03_formatted_polarity_prabhath.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VoDQFkoYOPOlfJjr8YOSMOK86QSbpOM3
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install torchvision
import torch
import torchvision

train_on_gpu = torch.cuda.is_available()
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ... To use GPU, go under edit > notebook settings')
else:
    print('CUDA is available!  Training on GPU ...')
    print(gpu_info)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import re
import nltk
nltk.download('stopwords')
import time
import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import LancasterStemmer
from sklearn.utils import shuffle


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

train_df = pd.read_csv('/content/drive/MyDrive/257_Project/train.csv', encoding="ISO-8859-1", header=None)
train_df.columns = ['polarity', 'id', 'date', 'query', 'user', 'tweet']

test_df = pd.read_csv('/content/drive/MyDrive/257_Project/test.csv', encoding="ISO-8859-1", header=None)
test_df.columns = ['polarity', 'id', 'date', 'query', 'user', 'tweet']

train_df.shape

test_df.shape

word_bank = []

# Function to remove predefined stopwords to reduce disk usage
def preprocess(text):
    review = re.sub('[^a-zA-Z]',' ',text) 
    review = review.lower()
    review = review.split()
    ps = LancasterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    return ' '.join(review)

"""Training the model on half dataset for this milestone, will do it on total data for final submission

"""

train_df = shuffle(train_df,random_state=2)
train_df = train_df[1:800000]

"""Setting train set and test set in a similar format for easier processing."""

train_df['polarity'].value_counts()

train_df['polarity'] = train_df['polarity'].replace(4,1)
train_df

test_df

test_df['polarity'] = test_df['polarity'].replace(2,1)
test_df

test_df['polarity'] = test_df['polarity'].replace(4,1)
test_df

X_train = train_df['tweet'].apply(lambda x: preprocess(x))

y_train = train_df['polarity']
le = LabelEncoder()
y = le.fit_transform(y_train)

X_test = test_df['tweet']
y_test = test_df['polarity']

tfidf = TfidfVectorizer(max_features = 600)
X_train_tf = tfidf.fit_transform(X_train).toarray() 
X_test = tfidf.transform(X_test).toarray()

X_train_tf.shape, X_test.shape, y_train.shape, y_test.shape

"""**Logistic Regreession**"""

lr = LogisticRegression(random_state = 0)
start_time = time.time()
lr.fit(X_train_tf, y_train) 
print("Execution Time:", time.time()-start_time,"secs")

y_pred_lr = lr.predict(X_test)
print("Accuracy:\n", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

"""Current accuracy of the model using logistic regression on 400k samples : ~74%

**Decision Tree Classifier**
"""

dc = DecisionTreeClassifier(criterion = 'entropy', random_state = 22)
start_time = time.time()
dc.fit(X_train_tf, y_train)
print("Execution Time:", time.time()-start_time,"secs")

y_pred_dc = dc.predict(X_test)
print("Accuracy:\n", accuracy_score(y_test, y_pred_dc))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dc))
print("Classification Report:\n", classification_report(y_test, y_pred_dc))

"""Current accuracy of the model using decision tree classifier on 400k samples : ~65%

**Naive Bayes Classifier**
"""

nb = MultinomialNB()
start_time = time.time()
nb.fit(X_train_tf,y_train)
print("Execution Time:", time.time()-start_time,"secs")

y_pred_nb = nb.predict(X_test)
print("Accuracy:\n", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

"""Current accuracy of the model using decision tree classifier on 400k samples : ~72%"""