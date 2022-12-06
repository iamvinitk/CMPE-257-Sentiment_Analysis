import matplotlib
import numpy as np
import pandas as pd
import pyLDAvis
import tomotopy as tp
from nltk.corpus import stopwords
from wordcloud import WordCloud

print(tp.isa)

mdl = tp.PAModel(tw=tp.TermWeight.ONE, min_cf=0, min_df=0, rm_top=5, k1=5, k2=10, alpha=0.1, subalpha=0.1, eta=0.01,
                 seed=3, corpus=None, transform=None)

data = pd.read_csv("./cleaned_train.csv")

# remove data with no text

# remove single character words
data['tweet'] = data['tweet'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')

data = data[data['tweet'].notna()]

print(data.head())
for i in range(len(data)):
    # add tweets to model
    mdl.add_doc(data.iloc[i]['tweet'].split())

for i in range(0, 100, 20):
    mdl.train(20)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

# save model
mdl.save("./pam_model3.bin")

# load model
mdl = tp.PAModel.load("./pam_model3.bin")


# show word cloud for each topic
for k in range(mdl.k1):
    matplotlib.pyplot.figure(figsize=(10, 10))
    STOPWORDS = set(stopwords.words('english'))
    wordcloud = WordCloud(min_font_size=5, max_words=300, width=1600 , height=800 , stopwords=STOPWORDS).generate(
        ' '.join([w for w, _ in mdl.get_topic_words(k, top_n=100)]))
    matplotlib.pyplot.imshow(wordcloud, interpolation='bilinear')
    matplotlib.pyplot.axis("off")
    matplotlib.pyplot.show()


for k in range(mdl.k):
    print('Top 10 words of topic #{}'.format(k))
    for w in mdl.get_topic_words(k, top_n=10):
        print(w)
    print()

# save topics
topics = []
for k in range(mdl.k):
    topics.append(mdl.get_topic_words(k, top_n=10))

topics = pd.DataFrame(topics)

topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
vocab = list(mdl.used_vocabs)
term_frequency = mdl.used_vocab_freq
# %%
prepared_data = pyLDAvis.prepare(
    topic_term_dists,
    doc_topic_dists,
    doc_lengths,
    vocab,
    term_frequency,
    start_index=0,  # tomotopy starts topic ids with 0, pyLDAvis with 1
    sort_topics=False  # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pyLDAvis
import tomotopy as tp

# Frequency Distribution of Word Counts in Documents in the model
doc_lengths = [len(doc.words) for doc in mdl.docs]
sns.distplot(doc_lengths)

# Word Clouds of Top N Keywords in Each Topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=STOPWORDS,
                    background_color='white',
                    width=2500,
                    height=1800,
                    max_words=10,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: cols[i],
                    prefer_horizontal=1.0)

topics = mdl.get_topic_words(k=10, top_n=10)

fig, axes = plt.subplots(2, 5, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
