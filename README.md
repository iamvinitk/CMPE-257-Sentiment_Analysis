# Project - Sentiment Analysis - CMPE 257

## Team Members

- [Prabhath Reddy Gujavarthy](https://github.com/prabhath-r)
- [Reshma Chowdary Bobba](https://github.com/ReshmaBC)
- [Sathvick Reddy Narahari](https://github.com/SathvickN)
- [Vinit Kanani](https://www.github.com/iamvinitk)

## 1. Dataset
### 1.1. Data Source

[Sentiment140](http://help.sentiment140.com/for-students/)

### 1.2. Data Description
We are using Sentiment140 as a dataset. This dataset consists of tweets which are collected using twitter API, they labeled a tweet as positive if it has :) emoticon in it and negative if it has :( emoticon in it. The length of the training dataset is 16,00,000. The dataset consists of 6 columns, 0 - polarity of the tweet, 1 - id of the tweet, 2 - date of the tweet, 3 - the query, 4 - the user that tweeted, 5 - the text of the tweet. The polarity of the tweet column has three values they are 0 = neutral, 2 = negative and 4 = positive .
## 2. Problem Statement
We would like to perform sentiment analysis on the dataset . The data has a column called polarity which tells us if the tweet is neutral, negative or positive. We would like to see if the models we create to perform sentiment analysis are aligning with their assumption that tweets which have :) emoticon contain positive sentiment and tweets which have :( emoticon contain negative sentiment.We would like to perform topic modeling on all the tweets, which will help us in detecting the topics present in the tweets. This will help us to organize and also summarize such a large tweets dataset.


## 3. Methodology
We are planning to use supervised techniques for sentiment analysis of the tweets and unsupervised techniques for topic modeling. We are planning to use supervised techniques, Naive Bayes classifier and Decision Trees which will be trained on the labeled data to predict the sentiment of the test data. We are using Latent Dirichlet Allocation (LDA) an unsupervised technique to detect topics present in the tweets.


## 4. Results



This is an example project to build academic papers using markdown and pandoc.

## Install

First you need to install [Pandoc](https://pandoc.org/installing.html)

Next, install some Python packages:

	pip install -r requirements.txt

Finally, it is not necessary but installing the entr tool makes it so your pdf gets generated every time your markdown files change.


## Run

Modify paper.md then in a terminal window run either make or run.sh (auto build)