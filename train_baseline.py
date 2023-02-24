import re

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import math

import nltk

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

datapath = "taskset/"

data = pd.read_csv(datapath + 'twi_train.tsv', sep='\t', header=0)

nltk.download('wordnet')
nltk.download('omw-1.4')

dev_gold = pd.read_csv(datapath + 'twi_dev_gold_label.tsv', sep='\t', header=0)

dev = pd.read_csv(datapath + 'twi_dev.tsv', sep='\t', header=0)


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st
data['tweet'] = data.tweet.apply(lemmatize_text)

dev['tweet'] = dev.tweet.apply(lemmatize_text)


tweets = data['tweet'].values
labels = data['label'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)


dev_tweets = dev['tweet'].values
dev_labels = dev_gold['label'].values
encoder = LabelEncoder()
dev_encoded_labels = encoder.fit_transform(dev_labels)


from typing_extensions import Counter
m = Counter(dev_labels)


train_sentences, test_sentences, train_labels, test_labels = train_test_split(tweets, encoded_labels, stratify = encoded_labels)


def laplace_smoothing(n_label_items, vocab, word_counts, word, text_label):
    a = word_counts[text_label][word] + 1
    b = n_label_items[text_label] + len(vocab)
    return math.log(a/b)


def group_by_label(x, y, labels):
    data = {}
    for l in labels:
        data[l] = x[np.where(y == l)]
    return data


def fit(x, y, labels):
    n_label_items = {}
    log_label_priors = {}
    n = len(x)
    grouped_data = group_by_label(x, y, labels)
    for l, data in grouped_data.items():
        n_label_items[l] = len(data)
        log_label_priors[l] = math.log(n_label_items[l] / n)
    return n_label_items, log_label_priors


def predict(n_label_items, vocab, word_counts, log_label_priors, labels, x):
    result = []
    for text in x:
        label_scores = {l: log_label_priors[l] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in vocab: continue
            for l in labels:
                log_w_given_l = laplace_smoothing(n_label_items, vocab, word_counts, word, l)
                label_scores[l] += log_w_given_l
        result.append(max(label_scores, key=label_scores.get))
    return result


# prediction


vec = CountVectorizer(max_features = 3000)
X = vec.fit_transform(dev_tweets)
vocab = vec.get_feature_names()
X = X.toarray()
word_counts = {}
for l in range(3):
    word_counts[l] = defaultdict(lambda: 0)
for i in range(X.shape[0]):
    l = dev_encoded_labels[i]
    # print(l)
    for j in range(len(vocab)):
        word_counts[l][vocab[j]] += X[i][j]

labels = [0,1, 2]
n_label_items, log_label_priors = fit(train_sentences,train_labels,labels)
pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test_sentences)
print("Accuracy of prediction on eval set : ", accuracy_score(test_labels,pred))



n_label_items, log_label_priors = fit(train_sentences,train_labels,labels)
pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, dev_tweets)
print("Accuracy of prediction on dev set : ", accuracy_score(dev_encoded_labels,pred))