"""
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is cool)
"""

# read the data using pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import contractions

# read the data
data = pd.read_csv('data/training.csv', encoding="ISO-8859-1", header=None)
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# check the data
print(data.head())

# check the shape of the data
print(data.shape)

# check the distribution of the data
print(data.groupby('sentiment').size())

# check the distribution of the data
sns.countplot(data['sentiment'], label='Count', hue_order=[0, 4])
plt.show()

# check the number of missing values
print("Missing Values", data.isnull().sum())

# count the text field for http links
print("Number of http links", data['text'].str.count('http').sum())

# remove the http links
data['text'] = data['text'].str.replace('http\S+|www.\S+', '', case=False)

# count the text field for @ mentions
print("Number of @ mentions", data['text'].str.count('@').sum())

# remove the @ mentions
data['text'] = data['text'].str.replace('@\S+', '', case=False)

# print the first 5 rows of the data
print(data.head())

# convert all the short forms to their full forms

data['text'] = data['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])

# print the first 5 rows of the data
print(data.head())
