# """
# 0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
# 1 - the id of the tweet (2087)
# 2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
# 3 - the query (lyx). If there is no query, then this value is NO_QUERY.
# 4 - the user that tweeted (robotickilldozr)
# 5 - the text of the tweet (Lyx is cool)
# """
# from wordcloud import WordCloud
# from textblob import TextBlob
# # read the data using pandas
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import contractions
# import numpy as np
#
# # read the data
# data = pd.read_csv('data/training.csv', encoding="ISO-8859-1", header=None)
# data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
#
# # check the data
# print(data.head())
#
# # check the shape of the data
# print(data.shape)
#
# # check the distribution of the data
# print(data.groupby('sentiment').size())
#
# # check the distribution of the data
# sns.countplot(data['sentiment'], label='Count', hue_order=[0, 4])
# plt.show()
#
# # check the number of missing values
# print("Missing Values", data.isnull().sum())
#
# # count the text field for http links
# print("Number of http links", data['text'].str.count('http').sum())
#
# # remove the http links
# data['text'] = data['text'].str.replace(r'http\S+|www.\S+', '', case=False, regex=True)
#
# # count the text field for @ mentions
# print("Number of @ mentions", data['text'].str.count('@').sum())
#
# # remove the @ mentions
# data['text'] = data['text'].str.replace(r'@\S+', '', case=False, regex=True)
#
# # count the text field for # mentions
# print("Number of # mentions", data['text'].str.count('#').sum())
#
# # remove the # tags
# data['text'] = data['text'].str.replace(r'#\S+', '', case=False, regex=True)
#
# # count the text field for RT
# print("Number of RT", data['text'].str.count('RT').sum())
#
# # remove the RT
# data['text'] = data['text'].str.replace(r'RT', '', case=False, regex=True)
#
# # calculate the polarity of the tweet using TextBlob
# from textblob import TextBlob
#
#
# # create a function to calculate the polarity
# def get_polarity(text):
#     return TextBlob(text).sentiment.polarity
#
#
# # create a new column for the polarity
# data['polarity'] = data['text'].apply(get_polarity)
#
# # plot the polarity
# sns.distplot(data['polarity'])
#
#
# # Visualize the word cloud
#
#
# # create a function to get the word cloud
# def get_word_cloud(text):
#     word_cloud = WordCloud().generate(text)
#     plt.imshow(word_cloud)
#     plt.axis("off")
#     plt.show()
#
#
# # get the word cloud for the positive tweets
# get_word_cloud(data[data['sentiment'] == 4]['text'].str.cat(sep=' '))
# # get the word cloud for the negative tweets
# get_word_cloud(data[data['sentiment'] == 0]['text'].str.cat(sep=' '))
#
# # remove the punctuation
# data['text'] = data['text'].str.replace(r'[^\w\s]', '', case=False, regex=True)
#
# # compare the polarity of the tweets and the sentiment
# sns.boxplot(x='sentiment', y='polarity', data=data)
#
# # replace the sentiment with -1 and 1
# data['sentiment'] = data['sentiment'].replace(0, -1)
# data['sentiment'] = data['sentiment'].replace(4, 1)
#
# # confusion matrix of the polarity and the sentiment
# from sklearn.metrics import confusion_matrix
#
#
# # create a function to get the confusion matrix
# def get_confusion_matrix(data, actual_column, predicted_column):
#     cm = confusion_matrix(data[actual_column], data[predicted_column], normalize='true')
#     # add the labels in terms of percentage
#     labels = [f'{v:.2%}' for v in cm.flatten()]
#     labels = np.asarray(labels).reshape(2, 2)
#     sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
#     plt.show()
#
#
# get_confusion_matrix(data, 'sentiment', 'polarity')
# # get the confusion matrix
#
# # update the polarity to -1 if less than -.25 and 1 if greater than .25
# data['polarity'] = data['polarity'].apply(lambda x: -1 if x < -.25 else 1 if x > .25 else 0)
# data['polarity_num'] = data['polarity'].apply(lambda x: 1 if x > 0 else -1)
#
# pd.set_option('display.max_colwidth', None)
#
# # plot histogram of the polarity
# sns.distplot(data['polarity_num'])
#
# # compare the sentiment against spacy polarity
# import spacy
#
# # load the spacy model
# nlp = spacy.load('en_core_web_sm')
#
# # add the spacy text blob to the pipeline
# nlp.add_pipe('spacytextblob')
#
#
# def get_polarity_spacy(text):
#     doc = nlp(text)
#     return doc._.polarity
#
#
# # create a new column for the polarity
# data['polarity_spacy'] = data['text'].apply(get_polarity_spacy)
#
# # compare the sentiment against nltk polarity
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
#
# # load the nltk model
# nltk_model = SentimentIntensityAnalyzer()
#
#
# # create a function to get the polarity
# def get_polarity_nltk(text):
#     return nltk_model.polarity_scores(text)['compound']
#
#
# data['polarity_nltk'] = data['text'].apply(get_polarity_nltk)
#
# # drop all the rows with sentiment not equal to polarity
# data = data[data['sentiment'] == data['polarity_num']]
# # drop all the rows with sentiment not equal to polarity
# data = data[data['sentiment'] == data['polarity_spacy']]
#
# sns.displot(data['polarity_nltk'])
#
# # print percentage of the wrong polarity
# print("Percentage of wrong polarity", len(data[data['sentiment'] != data['polarity_nltk']]) / len(data))