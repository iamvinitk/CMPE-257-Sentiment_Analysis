import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

data = pd.read_csv('./code/app/data/training.csv', encoding="ISO-8859-1", header=None)
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'tweet']

print(data.head())

print("Size of the dataset", data.shape)

print("Missing Values \n\n", data.isnull().sum())

# Preprocessing
print("Number of http links", data['tweet'].str.count('http').sum())
data['tweet'] = data['tweet'].str.replace(r'http\S+|www.\S+', '', case=False, regex=True)

print("Number of @ mentions", data['tweet'].str.count('@').sum())
data['tweet'] = data['tweet'].str.replace(r'@\S+', '', case=False, regex=True)

print("Number of # mentions", data['tweet'].str.count('#').sum())
data['tweet'] = data['tweet'].str.replace(r'#\S+', '', case=False, regex=True)

print("Number of RT", data['tweet'].str.count('RT').sum())
data['tweet'] = data['tweet'].str.replace(r'RT', '', case=False, regex=True)

stop_words = set(stopwords.words('english'))
stop_words.add('quot')
stop_words.add('amp')

lemma = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text)
    text = [item for item in text if item not in stop_words]
    text = [lemma.lemmatize(w) for w in text]
    text = [i for i in text if len(i) > 2]
    text = ' '.join(text)
    return text


data['clean_tweet'] = data['tweet'].apply(clean_text)
