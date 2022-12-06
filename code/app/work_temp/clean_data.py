# import re
# import swifter
import pandas as pd

# from nltk import WordNetLemmatizer, word_tokenize
# from nltk.corpus import stopwords
# from nltk.sentiment import SentimentIntensityAnalyzer
# from sklearn import preprocessing
# from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import f1_score
# from sklearn.svm import SVC
# import multiprocessing as mp
# from flair.models import TextClassifier
# from flair.data import Sentence
#
# sia = TextClassifier.load('en-sentiment')
#
# stop_words = set(stopwords.words('english'))
# stop_words.add('quot')
# stop_words.add('amp')
#
# lemma = WordNetLemmatizer()
#
# # print complete dataframe
# pd.set_option('display.max_columns', None)
#
#
# # remove continuously repeated words
# def remove_repeated_words(text):
#     try:
#         pattern = re.compile(r"\b(\w+)( \1\b)+")
#         return pattern.sub(r"\1", text)
#     except Exception as e:
#         print(e, text)
#         return text
#
#
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'http\S+', ' ', text)
#     text = re.sub('[^a-zA-Z]', ' ', text)
#     text = word_tokenize(text)
#     text = [item for item in text if item not in stop_words]
#     text = [lemma.lemmatize(w) for w in text]
#     text = [i for i in text if len(i) > 2]
#     text = ' '.join(text)
#     return text
#
#
# def remove_repeated_characters(text):
#     try:
#         pattern = re.compile(r"(.)\1{4,}", re.DOTALL)
#         return pattern.sub(r"\1\1", text)
#     except Exception as e:
#         print(e, text)
#         return text
#
#
# def clean_test():
#     data = pd.read_csv('../data/test.csv', encoding="ISO-8859-1", header=None)
#     data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'tweet']
#
#     print(data.head())
#
#     print("Number of http links", data['tweet'].str.count('http').sum())
#     data['tweet'] = data['tweet'].str.replace(r'http\S+|www.\S+', '', case=False, regex=True)
#
#     print("Number of @ mentions", data['tweet'].str.count('@').sum())
#     data['tweet'] = data['tweet'].str.replace(r'@\S+', '', case=False, regex=True)
#
#     print("Number of # mentions", data['tweet'].str.count('#').sum())
#     data['tweet'] = data['tweet'].str.replace(r'#\S+', '', case=False, regex=True)
#
#     print("Number of RT", data['tweet'].str.count('RT').sum())
#     data['tweet'] = data['tweet'].str.replace(r'RT', '', case=False, regex=True)
#
#     data['clean_tweet'] = data['tweet'].apply(clean_text)
#
#     data['clean_tweet'] = data['clean_tweet'].apply(remove_repeated_characters)
#
#     data['clean_tweet'] = data['clean_tweet'].apply(remove_repeated_words)
#
#     # drop tweet column
#     data.drop('tweet', axis=1, inplace=True)
#
#     # rename clean_tweet to tweet
#     data.rename(columns={'clean_tweet': 'tweet'}, inplace=True)
#
#     # change sentiment to -1, 0, 1
#     data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 4 else x)
#     data['sentiment'] = data['sentiment'].apply(lambda x: -1 if x == 0 else x)
#     data['sentiment'] = data['sentiment'].apply(lambda x: 0 if x == 2 else x)
#     # save the cleaned data
#     data = data[['id', 'tweet', 'sentiment']]
#
#     data.to_csv('./reduced_test.csv', index=False)
#
#
# def clean_train():
#     data = pd.read_csv('../data/training.csv', encoding="ISO-8859-1", header=None)
#     data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'tweet']
#
#     print("Number of http links", data['tweet'].str.count('http').sum())
#     data['tweet'] = data['tweet'].str.replace(r'http\S+|www.\S+', '', case=False, regex=True)
#
#     print("Number of @ mentions", data['tweet'].str.count('@').sum())
#     data['tweet'] = data['tweet'].str.replace(r'@\S+', '', case=False, regex=True)
#
#     print("Number of # mentions", data['tweet'].str.count('#').sum())
#     data['tweet'] = data['tweet'].str.replace(r'#\S+', '', case=False, regex=True)
#
#     print("Number of RT", data['tweet'].str.count('RT').sum())
#     data['tweet'] = data['tweet'].str.replace(r'RT', '', case=False, regex=True)
#
#     data['clean_tweet'] = data['tweet'].apply(clean_text)
#
#     data['clean_tweet'] = data['clean_tweet'].apply(remove_repeated_characters)
#
#     data['clean_tweet'] = data['clean_tweet'].apply(remove_repeated_words)
#
#     # drop tweet column
#     data.drop('tweet', axis=1, inplace=True)
#
#     # rename clean_tweet to tweet
#     data.rename(columns={'clean_tweet': 'tweet'}, inplace=True)
#
#     sid = SentimentIntensityAnalyzer()
#     data['polarity_vader'] = data['tweet'].apply(lambda x: sid.polarity_scores(x)['compound'])
#
#     data = data.sort_values(by=['polarity_vader'], ascending=False)
#
#     # save the cleaned data
#     data.to_csv('./cleaned_train_polarity.csv', index=False)
#
#
# # i = 0
# #
# #
# # def flair_prediction(x):
# #     global i
# #     i += 1
# #     if i % 100 == 0:
# #         print(i)
# #     sentence = Sentence(x)
# #     sia.predict(sentence)
# #     score = sentence.score
# #     return score
#
#
# def reduce_train():
#     data = pd.read_csv('./cleaned_train.csv')
#     # get the top 10000 rows
#     # with mp.Pool(mp.cpu_count()) as pool:
#     #     data["polarity_flair"] = pool.map(flair_prediction, data["tweet"])
#     data['polarity_flair'] = data['tweet'].swifter.apply(lambda x: flair_prediction(x))
#     sid = SentimentIntensityAnalyzer()
#     data['polarity_vader'] = data['tweet'].apply(lambda x: sid.polarity_scores(x)['compound'])
#
#     # remove missing values
#     data = data.dropna()
#
#     # print(data.head())
#     # # get 1000 rows for top and bottom
#     # top = data[data['polarity_flair'] > 0.40].sample(n=50000)
#     # bottom = data[-1000:]
#     # bottom = data[data['polarity_flair'] < -0.40].sample(n=50000)
#     #
#     # # get 1000 rows where polarity_vader is 0
#     # neutral = data[(data['polarity_flair'] < 0.40)]
#     # neutral = neutral[(neutral['polarity_flair'] > -0.40)]
#     # neutral = neutral.sample(n=50000)
#     # print(neutral.head())
#     #
#     # # combine all the data
#     # data = pd.concat([top, neutral, bottom])
#     # print(data.shape)
#     #
#     # # drop sentiment column
#     # data.drop('sentiment', axis=1, inplace=True)
#     #
#     # # rename polarity_vader to sentiment
#     # data.rename(columns={'polarity_flair': 'sentiment'}, inplace=True)
#     #
#     # # change sentiment to -1, 0, 1
#     # data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x > 0.35 else (-1 if x < -0.35 else 0))
#     # save to file
#     data = data[['id', 'tweet', 'sentiment', 'polarity_flair', 'polarity_vader']]
#     data.to_csv('./cleaned_train_labelled.csv', index=False)
#
#
# def main():
#     train_data = pd.read_csv('../data/reduced_train.csv')
#     test_data = pd.read_csv('../data/reduced_test.csv')
#     print(test_data.head())
#
#     encoded_labels = preprocessing.LabelEncoder()
#
#     train_labels = encoded_labels.fit_transform(train_data['sentiment'])
#     test_labels = encoded_labels.fit_transform(test_data['sentiment'])
#
#     # TF-IDF
#     tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
#                             token_pattern=r'\w{1,}',
#                             ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
#
#     tfidf.fit(list(train_data['tweet']) + list(test_data['tweet']))
#
#     train_features = tfidf.transform(train_data['tweet'])
#     test_features = tfidf.transform(test_data['tweet'])
#
#     print("==>", train_features.shape)
#     print("==>", test_features.shape)
#
#     # SVD
#     svd = TruncatedSVD(n_components=300)
#     svd.fit(train_features)
#
#     train_features = svd.transform(train_features)
#     test_features = svd.transform(test_features)
#
#     # Normalize data with StandardScaler
#     scaler = preprocessing.StandardScaler()
#     scaler.fit(train_features)
#
#     train_features = scaler.transform(train_features)
#     test_features = scaler.transform(test_features)
#
#     # SVC model
#     svc = SVC(C=1.0, probability=True)
#
#     svc.fit(train_features, train_labels)
#
#     # predict
#     predictions = svc.predict(test_features)
#
#     # calculate f1 score
#     f1 = f1_score(test_labels, predictions, average='weighted')
#     print(f1)
#
#
# def main2():
#     train_data = pd.read_csv('../data/reduced_train.csv')
#     test_data = pd.read_csv('../data/reduced_test.csv')
#
#     # split data into train and test
#     # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#
#     encoded_labels = preprocessing.LabelEncoder()
#
#     train_labels = encoded_labels.fit_transform(train_data['sentiment'])
#     test_labels = encoded_labels.fit_transform(test_data['sentiment'])
#
#     # TF-IDF
#     tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
#                             token_pattern=r'\w{1,}',
#                             ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
#
#     tfidf.fit(list(train_data['tweet']) + list(test_data['tweet']))
#
#     train_features = tfidf.transform(train_data['tweet'])
#     test_features = tfidf.transform(test_data['tweet'])
#
#     print("==>", train_features.shape)
#     print("==>", test_features.shape)
#
#     # SVD
#     svd = TruncatedSVD(n_components=300)
#     svd.fit(train_features)
#
#     train_features = svd.transform(train_features)
#     test_features = svd.transform(test_features)
#
#     # Normalize data with StandardScaler
#     scaler = preprocessing.StandardScaler()
#     scaler.fit(train_features)
#
#     train_features = scaler.transform(train_features)
#     test_features = scaler.transform(test_features)
#
#     # SVC model
#     svc = SVC(C=1.0, probability=True)
#
#     svc.fit(train_features, train_labels)
#
#     # predict
#     predictions = svc.predict(test_features)
#
#     # calculate f1 score
#     f1 = f1_score(test_labels, predictions, average='weighted')
#     print(f1)
#
#
# # clean_test()
# # clean_train()
#
# # reduce_train()
#
# # main()
#
#
# data = pd.read_csv('./cleaned_train.csv')
# print(data.head())
#
# i = 0
#
#
# def flair_prediction(x):
#     x = str(x)
#     try:
#         global i
#         i += 1
#         if i % 10000 == 0:
#             print(i)
#         sentence = Sentence(x)
#         sia.predict(sentence)
#         score = sentence.score
#         return score
#     except Exception as e:
#         return 0
#
#
# with mp.Pool(mp.cpu_count()) as pool:
#     data['polarity_flair'] = pool.map(flair_prediction, data['tweet'])
# data = data[['id', 'tweet', 'sentiment', 'polarity_flair', 'polarity_vader']]
# data.to_csv('./cleaned_train_labeled.csv', index=False)
data = pd.read_csv('./cleaned_train.csv')
print(data.columns)
print(data.shape)
data = data.dropna()
# drop polarity_vader column
data = data.drop(['polarity_vader'], axis=1)
print(data.shape)
data.to_csv('../data/cleaned_train.csv', index=False)

s, subplots = plt.subplots(2, 1)
plt.tight_layout()

# add a margin between subplots
s.subplots_adjust(hspace=0.5)

print(subplots.shape)
subplots[0].plot(history.history['accuracy'], c='b')
subplots[0].plot(history.history['val_accuracy'], c='r')
subplots[0].set_title('Accuracy')
subplots[0].set_ylabel('accuracy')
subplots[0].set_xlabel('epochs')
subplots[0].legend(['Train LSTM ', 'Value LSTM'], loc='upper left')

subplots[1].plot(history.history['loss'], c='m')
subplots[1].plot(history.history['val_loss'], c='c')
subplots[1].set_title('Loss')
subplots[1].set_ylabel('loss')
subplots[1].set_xlabel('epoch')
subplots[1].legend(['Train', 'Value'], loc='upper left')