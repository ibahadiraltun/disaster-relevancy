import spacy
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

ner_list = nlp.pipe_labels['ner']
pos_list = nlp.pipe_labels['tagger']

df = pd.read_csv('train_data.csv')
print(df.head())


def clean_tweet(tweet):
    # if type(tweet) == float:
    #     return ""
    temp = tweet.lower()
    # temp = re.sub("'", "", temp) # to avoid removing contractions in english
    # temp = re.sub("@[A-Za-z0-9_]+","", temp)
    # temp = re.sub("#[A-Za-z0-9_]+","", temp)
    # temp = re.sub(r"http\S+", "", temp)
    # temp = re.sub(r"www.\S+", "", temp)
    # temp = re.sub('[()!?]', ' ', temp)
    # temp = re.sub('\[.*?\]',' ', temp)
    # temp = re.sub("[^a-z0-9]"," ", temp)
    # temp = tweet.split()
    #
    # swords = stopwords.words('english')
    # temp = [w for w in temp if not w in swords]
    # temp = " ".join(word for word in temp)
    return temp


# tweets = [clean_tweet(tw) for tw in df['text']]
tweets = list(df['text'])
labels = list(df['target'])

# X_train = tweets
# X_test = []
# y_train = labels
# y_test = []

X_train,\
    X_test,\
    y_train,\
    y_test = train_test_split(tweets, labels, test_size=0.2, shuffle=True, random_state=1)


def vectorize_given_data(X, is_binary = False):
    res_vector = [[] for _ in range(len(X))]

    for idx in range(len(X)):
        tweet = X[idx]
        feat_text = nlp(tweet)

        for cur_ner in ner_list:
            current_ner_count = 0
            for cur_ent in feat_text.ents:
                if cur_ent.label_ == cur_ner:
                    if is_binary: current_ner_count = 1
                    else: current_ner_count = current_ner_count + 1
            res_vector[idx].append(current_ner_count)

        for cur_pos in pos_list:
            current_pos_count = 0
            for cur_token in feat_text:
                if cur_token.pos_ == cur_pos:
                    if is_binary: current_pos_count = 1
                    else: current_pos_count = current_pos_count + 1
            res_vector[idx].append(current_pos_count)

    return res_vector


X_train_vector = vectorize_given_data(X_train)
X_test_vector = vectorize_given_data(X_test)

from sklearn import linear_model

print('training model...')

logr = linear_model.LogisticRegression()
logr.fit(X_train_vector, y_train)

print('getting predictions on the test data...')

preds = []
probs = logr.predict_proba(X_test_vector)
print(probs)
for tmp in probs:
    if tmp[1] > 0.4:
        preds.append(1)
    else:
        preds.append(0)

score = logr.score(X_test_vector, y_test)
print('acc score -> ', score)

from sklearn import metrics
print('classification report -> ', metrics.classification_report(y_test, preds, digits=3))
print('confusion matrix -> ', metrics.confusion_matrix(y_test, preds))

# ### kaggle submission
#
# print('generating kaggle submission...')
#
# test_df = pd.read_csv('test_data.csv')
# origin_X_test = list(test_df['text'])
# origin_idx = list(test_df['id'])
#
# origin_X_test_vector = vectorize_given_data(origin_X_test)
# preds = logr.predict(origin_X_test_vector)
# submission_data = []
# for idx in range(len(preds)):
#     data_id = origin_idx[idx]
#     pred = preds[idx]
#     submission_data.append([data_id, pred])
#
# submission_df = pd.DataFrame(submission_data, columns=['id', 'target'])
# submission_df.to_csv('submission.csv', index=False)