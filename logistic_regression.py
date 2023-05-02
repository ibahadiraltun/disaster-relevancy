import pandas as pd
import numpy as np
import spacy
import re

# Load data
train = pd.read_csv("./train_data.csv")
test = pd.read_csv("./test_data.csv")
sub_sample = pd.read_csv("./sample_submission.csv")

print (train.shape, test.shape, sub_sample.shape)

train = train.drop_duplicates().reset_index(drop=True)

print(train)

x = input

train.isnull().sum()

test.isnull().sum()

""" Data cleaning """
test_str = train.loc[417, 'text']


def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'\n',' ', text)           # Remove line breaks
    text = re.sub('\s+', ' ', text).strip()  # Remove leading, trailing, and extra spaces
    return text


print("Original text: " + test_str)
print("Cleaned text: " + clean_text(test_str))


def find_hashtags(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'


def find_mentions(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'


def find_links(tweet):
    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'


def process_text(df):
    df['text_clean'] = df['text'].apply(lambda x: clean_text(x))
    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))
    df['mentions'] = df['text'].apply(lambda x: find_mentions(x))
    df['links'] = df['text'].apply(lambda x: find_links(x))
    return df


train = process_text(train)
test = process_text(test)

# train.corr()['target'].drop('target').sort_values()

""" Setting the features:
    1. count vector of the links
    2. count vector of the mentions
    3. count vector of the hastags
    4. TF-IDF vector of the tweet
    5. NER tags
    6. POS tags
"""

from sklearn.feature_extraction.text import CountVectorizer


# CountVectorizer

vec_loc = CountVectorizer(min_df = 1)
loc_vec = vec_loc.fit_transform(train['location'].apply(lambda x: np.str_(x)))
loc_vec_test = vec_loc.transform(test['location'].apply(lambda x: np.str_(x)))
X_train_loc = pd.DataFrame(loc_vec.toarray(), columns=vec_loc.get_feature_names_out())
X_test_loc = pd.DataFrame(loc_vec_test.toarray(), columns=vec_loc.get_feature_names_out())

# Links
vec_links = CountVectorizer(min_df = 5, analyzer = 'word', token_pattern = r'https?://\S+') # Only include those >=5 occurrences
link_vec = vec_links.fit_transform(train['links'])
link_vec_test = vec_links.transform(test['links'])
X_train_link = pd.DataFrame(link_vec.toarray(), columns=vec_links.get_feature_names_out())
X_test_link = pd.DataFrame(link_vec_test.toarray(), columns=vec_links.get_feature_names_out())

# Mentions
vec_men = CountVectorizer(min_df = 5)
men_vec = vec_men.fit_transform(train['mentions'])
men_vec_test = vec_men.transform(test['mentions'])
X_train_men = pd.DataFrame(men_vec.toarray(), columns=vec_men.get_feature_names_out())
X_test_men = pd.DataFrame(men_vec_test.toarray(), columns=vec_men.get_feature_names_out())

# Hashtags
vec_hash = CountVectorizer(min_df = 5)
hash_vec = vec_hash.fit_transform(train['hashtags'])
hash_vec_test = vec_hash.transform(test['hashtags'])
X_train_hash = pd.DataFrame(hash_vec.toarray(), columns=vec_hash.get_feature_names_out())
X_test_hash = pd.DataFrame(hash_vec_test.toarray(), columns=vec_hash.get_feature_names_out())
print (X_train_link.shape, X_train_men.shape, X_train_hash.shape)

_ = (X_train_link.transpose().dot(train['target']) / X_train_link.sum(axis=0)).sort_values(ascending=False)

_ = (X_train_men.transpose().dot(train['target']) / X_train_men.sum(axis=0)).sort_values(ascending=False)

hash_rank = (X_train_hash.transpose().dot(train['target']) / X_train_hash.sum(axis=0)).sort_values(ascending=False)
print('Hashtags with which 100% of Tweets are disasters: ')
print(list(hash_rank[hash_rank==1].index))
print('Total: ' + str(len(hash_rank[hash_rank==1])))
print('Hashtags with which 0% of Tweets are disasters: ')
print(list(hash_rank[hash_rank==0].index))
print('Total: ' + str(len(hash_rank[hash_rank==0])))

# Tf-idf for text
from sklearn.feature_extraction.text import TfidfVectorizer

vec_text = TfidfVectorizer(min_df = 10, ngram_range = (1,2), stop_words='english')
# Only include >=10 occurrences
# Have unigrams and bigrams
text_vec = vec_text.fit_transform(train['text_clean'])
text_vec_test = vec_text.transform(test['text_clean'])
X_train_text = pd.DataFrame(text_vec.toarray(), columns=vec_text.get_feature_names_out())
X_test_text = pd.DataFrame(text_vec_test.toarray(), columns=vec_text.get_feature_names_out())
print (X_train_text.shape)

# NER and POS Tags (NP)

nlp = spacy.load("en_core_web_sm")

ner_list = ['ORG', 'PERSON', 'MONEY', 'GPE', 'LOC', 'DATE', 'TIME']
pos_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET',
            'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ',
            'SYM', 'VERB', 'X', 'SPACE']

print(ner_list)
print(pos_list)

def ner_vectorize_given_data(X, is_binary = False):
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

    return res_vector


def pos_vectorize_given_data(X, is_binary = False):
    res_vector = [[] for _ in range(len(X))]

    for idx in range(len(X)):
        tweet = X[idx]
        feat_text = nlp(tweet)
        for cur_pos in pos_list:
            current_pos_count = 0
            for cur_token in feat_text:
                if cur_token.pos_ == cur_pos:
                    if is_binary: current_pos_count = 1
                    else: current_pos_count = current_pos_count + 1
            res_vector[idx].append(current_pos_count)

    return res_vector


# POS vectors
ner_vec = ner_vectorize_given_data(train['text_clean'])
ner_vec_test = ner_vectorize_given_data(test['text_clean'])
X_train_ner = pd.DataFrame(ner_vec, columns=ner_list)
X_test_ner = pd.DataFrame(ner_vec_test, columns=ner_list)

# NER vectors
pos_vec = pos_vectorize_given_data(train['text_clean'])
pos_vec_test = pos_vectorize_given_data(test['text_clean'])
X_train_pos = pd.DataFrame(pos_vec, columns=pos_list)
X_test_pos = pd.DataFrame(pos_vec_test, columns=pos_list)

# Joining the dataframes together

train = train.join(X_train_loc, rsuffix='_loc')
train = train.join(X_train_link, rsuffix='_link')
train = train.join(X_train_men, rsuffix='_mention')
train = train.join(X_train_hash, rsuffix='_hashtag')
train = train.join(X_train_text, rsuffix='_text')
train = train.join(X_train_ner, rsuffix='_ner')
train = train.join(X_train_pos, rsuffix='_pos')
test = test.join(X_test_loc, rsuffix='_loc')
test = test.join(X_test_link, rsuffix='_link')
test = test.join(X_test_men, rsuffix='_mention')
test = test.join(X_test_hash, rsuffix='_hashtag')
test = test.join(X_test_text, rsuffix='_text')
test = test.join(X_test_ner, rsuffix='_ner')
test = test.join(X_test_pos, rsuffix='_pos')
print (train.shape, test.shape)

"""## Logistic Regression"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

features_to_drop = ['id', 'keyword','text', 'location','text_clean', 'hashtags', 'mentions','links']
scaler = MinMaxScaler()

X_train = train.drop(columns = features_to_drop + ['target'])
X_test = test.drop(columns = features_to_drop)
y_train = train.target

print(X_train)

lr = LogisticRegression(solver='liblinear', random_state=777) # Other solvers have failure to converge problem

pipeline = Pipeline([('scale',scaler), ('lr', lr),])

pipeline.fit(X_train, y_train)
y_test = pipeline.predict(X_test)

submit = sub_sample.copy()
submit.target = y_test
submit.to_csv('submit_test.csv',index=False)

print ('Training accuracy: %.4f' % pipeline.score(X_train, y_train))

# F-1 score
from sklearn.metrics import f1_score

print ('Training f-1 score: %.4f' % f1_score(y_train, pipeline.predict(X_train)))

# Confusion matrix
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_train, pipeline.predict(X_train)))

