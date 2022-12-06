# read cleaned data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# data = pd.read_csv('./data/cleaned_data.csv')
# print(data.shape)
#
# # find polarity using vader
#
#
# sid = SentimentIntensityAnalyzer()
# data['polarity_vader'] = data['tweet'].apply(lambda x: sid.polarity_scores(x)['compound'])
#
# # sort data based on polarity_vader
# data = data.sort_values(by=['polarity_vader'], ascending=False)
#
# # group data based on polarity_vader
#
#
# # add polarity column with -1, 0, 1
# data['polarity_vader_score'] = data['polarity_vader'].apply(lambda x: 1 if x > 0.25 else (-1 if x < -0.25 else 0))
#
# data = data.groupby('polarity_vader_score').head(2000)
#
# # save data to file
# data.to_csv('./data/cleaned_data_vader.csv', index=False)

data = pd.read_csv('./temp')
# reset index
data = data.reset_index(drop=True)

# drop sentiment column
data = data.drop(['sentiment'], axis=1)

data['polarity_vader_score'] = data['polarity_vader'].apply(lambda x: 1 if x > 0.35 else (-1 if x < -0.35 else 0))

data = data.groupby('polarity_vader_score').head(2000)

# rename polarity_vader_score to sentiment
data = data.rename(columns={'polarity_vader_score': 'sentiment'})

# remove all columns except tweet, id and sentiment
data = data[['id', 'tweet', 'sentiment']]

# save data to file
data.to_csv('./data/cleaned_data_vader1.csv', index=False)
print(data.head())

# TODO - Perform comparison with different train/test
