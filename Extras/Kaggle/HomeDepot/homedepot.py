
# coding: utf-8

# AirBnB recruiting kaggle
# ------
#
# https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

# ### Load libraries

# Core
from __future__ import print_function
import datetime as DT
import pandas as pd
import numpy as np
import re
import logging
import sys
from itertools import product
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

# Sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler,Imputer
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, \
                                    ElasticNetCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

# distance metrics
from scipy.spatial.distance import *

# Metrics
from sklearn.metrics import log_loss, classification_report \
                            , label_ranking_average_precision_score, label_ranking_loss
from sklearn.metrics import make_scorer

# Neural Nets
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU

# XGBoost
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

# matplotlib
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

# nltk
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import *
#print all the synset element of an element
def synonyms(string):
    syndict = {}
    for i,j in enumerate(wn.synsets(string)):
        syns = j.lemma_names
        for syn in syns:
            syndict.setdefault(syn,1)
    keys = syndict.keys()
    out = [ k for k in keys if k != string ]
    return out

# custom functions
WORDS = re.compile(r'[a-zA-Z_]+')
stemmer = SnowballStemmer("english")
def findwords(x):
    words = ' '.join(WORDS.findall(str(x).lower()))
    return words

def getstems(x):
    stems = [ stemmer.stem(w) for w in x.split() ]
    combined = ' '.join(stems)
    return combined

findwords = np.vectorize(findwords)
getstems = np.vectorize(getstems)
synonym_text = np.vectorize(lambda x: ' '.join(' '.join(synonyms(z) if synonyms(z) else []) for z in set(x.split())))
countwords = np.vectorize(lambda x: len(WORDS.findall(str(x).lower())))

# ### Declare Args
def declare_args():
    ''' Declare global arguments and strings
    '''
    global ATTRIBUTES_FILE
    global PRODUCT_DESCRIPTIONS_FILE
    global SAMPLE_SUBMISSION_FILE
    global TEST_DATA_FINAL_FILE
    global TRAIN_DATA_FILE
    global TEST_SIZE
    global ATTRIBUTES_COLUMNS
    global PRODUCT_DESCRIPTIONS_COLUMNS
    global TRAIN_COLUMNS
    global TARGET_COLUMN
    global PARAMS

    ## Files ##
    ATTRIBUTES_FILE = 'Data/attributes.csv.gz'
    PRODUCT_DESCRIPTIONS_FILE = 'Data/product_descriptions.csv.gz'
    SAMPLE_SUBMISSION_FILE = 'Data/sample_submission.csv.gz'
    TEST_DATA_FINAL_FILE = 'Data/test.csv.gz'
    TRAIN_DATA_FILE = 'Data/train.csv.gz'

    ## Model args ##
    TEST_SIZE = 0.2

    ## Fields ##
    ATTRIBUTES_COLUMNS = [u'name', u'value']
    PRODUCT_DESCRIPTIONS_COLUMNS = [u'product_description']
    TRAIN_COLUMNS = []
    TARGET_COLUMN = [u'relevance']

    # regressor params
    PARAMS = {'n_estimators':1000, 'n_jobs':-1,'verbose':1,'max_features':15,'max_depth':20 }

# ### Read data
def load_data():
    ''' read in data files
    '''
    global train_full
    global target_full
    global final_test
    global attributes
    global descriptions
    logging.warn('Loading data files')

    ## Read training data
    train_full = pd.read_csv(TRAIN_DATA_FILE).sort_values('id')
    train_full.index = train_full['id']

    ## Read relevance data
    target_full = train_full[TARGET_COLUMN]

    ## Read in data to predict for submission ##
    final_test = pd.read_csv(TEST_DATA_FINAL_FILE)
    final_test.index = final_test['id']

    ## Read supplemental datasets ##
    attributes = pd.read_csv(ATTRIBUTES_FILE)
    attributes.index = attributes['product_uid']
    descriptions = pd.read_csv(PRODUCT_DESCRIPTIONS_FILE)
    descriptions.index = descriptions['product_uid']

def load_attributes(refresh=True, sourcefile='Data/attributes_new.csv'):
    global attributes
    logging.warn('Compiling attribute dimensions per product')
    if refresh:
        ## create matrix of attributes by product
        attributes['attribute_words'] = findwords(attributes['value'])
        del attributes['value']
        attributes['attribute_words'] = attributes['attribute_words'] + ' '
        attributes_new = pd.DataFrame(attributes['attribute_words'].groupby(level=0).sum())
        del attributes
        attributes_new['attribute_stems'] = getstems(synonym_text(attributes_new['attribute_words']))
        attributes_new.index = pd.Int64Index(attributes_new.index)
        attributes_new.to_csv(sourcefile, index=True)
        attributes = attributes_new
        del attributes_new
    else:
        attributes = pd.read_csv(sourcefile)

def load_descriptions(refresh=True, sourcefile='Data/descriptions_new.csv'):
    global descriptions
    logging.warn('Compiling descriptions into features for each product')
    if refresh:
        ## create matrix of descriptions by product
        descriptions['description_words'] = findwords(descriptions['product_description'])
        del descriptions['product_description']
        descriptions['description_stems'] = getstems(synonym_text(descriptions['description_words']))
        descriptions.index = descriptions['product_uid']
        descriptions.to_csv(sourcefile, index=True)
    else:
        descriptions = pd.read_csv(sourcefile)

def combine_data():
    global train_full
    global final_test
    logging.warn('Combining all datasets')

    ## filter out non-words in title
    train_full['product_title'] = findwords(train_full['product_title'])
    train_full['product_stems'] = getstems(synonym_text(train_full['product_title']))
    final_test['product_title'] = findwords(final_test['product_title'])
    final_test['product_stems'] = getstems(synonym_text(final_test['product_title']))

    ## Combine description data
    logging.warn('Combining descriptions')
    train_full.index = train_full['product_uid']
    final_test.index = final_test['product_uid']
    train_full = pd.merge(train_full, descriptions, how='left', left_index=True, right_index=True)
    final_test = pd.merge(final_test, descriptions, how='left', left_index=True, right_index=True)

    ## Combine attributes data
    logging.warn('Combining attributes')
    train_full = pd.merge(train_full, attributes, how='left', left_index=True, right_index=True)
    final_test = pd.merge(final_test, attributes, how='left', left_index=True, right_index=True)


    ## transform search terms
    logging.warn('Transforming search terms using stemmer and synonyms')
    train_full['search_term_original'] = train_full['search_term']
    final_test['search_term_original'] = final_test['search_term']
    train_full['search_term'] = findwords(train_full['search_term'])
    train_full['search_term'] = getstems(synonym_text(train_full['search_term']))
    final_test['search_term'] = findwords(final_test['search_term'])
    final_test['search_term'] = getstems(synonym_text(final_test['search_term']))

    ## Set ID's back and save data to disk
    train_full.index = train_full['id']
    final_test.index = final_test['id']
    train_full.to_csv('Data/train_full_stems.csv', index=True)
    final_test.to_csv('Data/final_test_stems.csv', index=True)

    ## vectorize words in attributes
    logging.warn('Calculate count and TF-IDF vectors for title+description+attributes')
    cv = CountVectorizer(stop_words='english')
    tf = TfidfTransformer()
    vocab = pd.concat((train_full['product_stems'].fillna(''),final_test['product_stems'].fillna(''))) \
                      +' '+pd.concat((train_full['description_stems'].fillna(''),final_test['description_stems'].fillna(''))) \
                      +' '+pd.concat((train_full['attribute_stems'].fillna(''),final_test['attribute_stems'].fillna('')))
    cv.fit(vocab)
    tf.fit(cv.transform(vocab))
    train_full_vec = cv.transform(train_full['product_stems'].fillna('')
                                    +' '+train_full['description_stems'].fillna('')
                                    +' '+train_full['attribute_stems'].fillna(''))
    final_test_vec = cv.transform(final_test['product_stems'].fillna('')
                                    +' '+final_test['description_stems'].fillna('')
                                    +' '+final_test['attribute_stems'].fillna(''))
    train_full_vec_tf = tf.transform(train_full_vec)
    final_test_vec_tf = tf.transform(final_test_vec)

    ## vectorize words in title
    logging.warn('Calculate count and TF-IDF vectors for title')
    cv_title = CountVectorizer(stop_words='english')
    tf_title = TfidfTransformer()
    train_full_vec_title = cv_title.fit_transform(train_full['product_stems'].fillna(''))
    final_test_vec_title = cv_title.transform(final_test['product_stems'].fillna(''))
    train_full_vec_tf_title = tf_title.fit_transform(train_full_vec_title)
    final_test_vec_tf_title = tf_title.transform(final_test_vec_title)

    ## vectorize words in description
    logging.warn('Calculate count and TF-IDF vectors for description')
    cv_desc = CountVectorizer(stop_words='english')
    tf_desc = TfidfTransformer()
    train_full_vec_desc = cv_desc.fit_transform(train_full['description_stems'].fillna(''))
    final_test_vec_desc = cv_desc.transform(final_test['description_stems'].fillna(''))
    train_full_vec_tf_desc = tf_desc.fit_transform(train_full_vec_desc)
    final_test_vec_tf_desc = tf_desc.transform(final_test_vec_desc)

    ## vectorize words in description
    logging.warn('Calculate count and TF-IDF vectors for attributes')
    cv_attr = CountVectorizer(stop_words='english')
    tf_attr = TfidfTransformer()
    train_full_vec_attr = cv_attr.fit_transform(train_full['attribute_stems'].fillna(''))
    final_test_vec_attr = cv_attr.transform(final_test['attribute_stems'].fillna(''))
    train_full_vec_tf_attr = tf_attr.fit_transform(train_full_vec_attr)
    final_test_vec_tf_attr = tf_attr.transform(final_test_vec_attr)

    ## vectorize search terms
    logging.warn('Calculate count and TF-IDF vectors for search terms')
    train_full_stvec = cv.transform(train_full['search_term'])
    final_test_stvec = cv.transform(final_test['search_term'])
    train_full_stvec_title = cv_title.transform(train_full['search_term'])
    final_test_stvec_title = cv_title.transform(final_test['search_term'])
    train_full_stvec_desc = cv_desc.transform(train_full['search_term'])
    final_test_stvec_desc = cv_desc.transform(final_test['search_term'])
    train_full_stvec_attr = cv_attr.transform(train_full['search_term'])
    final_test_stvec_attr = cv_attr.transform(final_test['search_term'])
    train_full_stvec_tf = tf.transform(train_full_stvec)
    final_test_stvec_tf = tf.transform(final_test_stvec)
    train_full_stvec_tf_title = tf_title.transform(train_full_stvec_title)
    final_test_stvec_tf_title = tf_title.transform(final_test_stvec_title)
    train_full_stvec_tf_desc = tf_desc.transform(train_full_stvec_desc)
    final_test_stvec_tf_desc = tf_desc.transform(final_test_stvec_desc)
    train_full_stvec_tf_attr = tf_attr.transform(train_full_stvec_attr)
    final_test_stvec_tf_attr = tf_attr.transform(final_test_stvec_attr)

    ## calculate distances between search terms and product data
    logging.warn('Run distance calculations')
    denseit = lambda x: np.array(x.todense()).ravel()
    N = train_full_vec.shape[0]
    N2 = final_test_vec.shape[0]
    metrics = ['euclidean','cosine','chebyshev','np.dot'
                ,'braycurtis','canberra','correlation','cityblock'
                ,'hamming','dice','rogerstanimoto'
                ,'sokalmichener','sokalsneath','sqeuclidean']
    distances_train = {}
    distances_test = {}
    for m in metrics:
        distances_train[m] = np.zeros(N); distances_test[m] = np.zeros(N2)
        distances_train[m+'_tf'] = np.zeros(N); distances_test[m+'_tf'] = np.zeros(N2)
        distances_train[m+'_title'] = np.zeros(N); distances_test[m+'_title'] = np.zeros(N2)
        distances_train[m+'_tf_title'] = np.zeros(N); distances_test[m+'_tf_title'] = np.zeros(N2)
        distances_train[m+'_desc'] = np.zeros(N); distances_test[m+'_desc'] = np.zeros(N2)
        distances_train[m+'_tf_desc'] = np.zeros(N); distances_test[m+'_tf_desc'] = np.zeros(N2)
        distances_train[m+'_attr'] = np.zeros(N); distances_test[m+'_attr'] = np.zeros(N2)
        distances_train[m+'_tf_attr'] = np.zeros(N); distances_test[m+'_tf_attr'] = np.zeros(N2)
    for m in metrics:
        logging.warn('Calculating distance metric {}'.format(m))
        for i in range(max(N,N2)):
            if i<N:
                distances_train[m][i] = eval(m+'(denseit(train_full_vec['+str(i)+',:]),denseit(train_full_stvec['+str(i)+',:]))')
                distances_train[m+'_tf'][i] = eval(m+'(denseit(train_full_vec_tf['+str(i)+',:]),denseit(train_full_stvec_tf['+str(i)+',:]))')
                distances_train[m+'_title'][i] = eval(m+'(denseit(train_full_vec_title['+str(i)+',:]),denseit(train_full_stvec_title['+str(i)+',:]))')
                distances_train[m+'_tf_title'][i] = eval(m+'(denseit(train_full_vec_tf_title['+str(i)+',:]),denseit(train_full_stvec_tf_title['+str(i)+',:]))')
                distances_train[m+'_desc'][i] = eval(m+'(denseit(train_full_vec_desc['+str(i)+',:]),denseit(train_full_stvec_desc['+str(i)+',:]))')
                distances_train[m+'_tf_desc'][i] = eval(m+'(denseit(train_full_vec_tf_desc['+str(i)+',:]),denseit(train_full_stvec_tf_desc['+str(i)+',:]))')
                distances_train[m+'_attr'][i] = eval(m+'(denseit(train_full_vec_attr['+str(i)+',:]),denseit(train_full_stvec_attr['+str(i)+',:]))')
                distances_train[m+'_tf_attr'][i] = eval(m+'(denseit(final_test_vec_tf_attr['+str(i)+',:]),denseit(train_full_stvec_tf_attr['+str(i)+',:]))')
            if i<N2:
                distances_test[m][i] = eval(m+'(denseit(final_test_vec['+str(i)+',:]),denseit(final_test_stvec['+str(i)+',:]))')
                distances_test[m+'_tf'][i] = eval(m+'(denseit(final_test_vec_tf['+str(i)+',:]),denseit(final_test_stvec_tf['+str(i)+',:]))')
                distances_test[m+'_title'][i] = eval(m+'(denseit(final_test_vec_title['+str(i)+',:]),denseit(final_test_stvec_title['+str(i)+',:]))')
                distances_test[m+'_tf_title'][i] = eval(m+'(denseit(final_test_vec_tf_title['+str(i)+',:]),denseit(final_test_stvec_tf_title['+str(i)+',:]))')
                distances_test[m+'_desc'][i] = eval(m+'(denseit(final_test_vec_desc['+str(i)+',:]),denseit(final_test_stvec_desc['+str(i)+',:]))')
                distances_test[m+'_tf_desc'][i] = eval(m+'(denseit(final_test_vec_tf_desc['+str(i)+',:]),denseit(final_test_stvec_tf_desc['+str(i)+',:]))')
                distances_test[m+'_attr'][i] = eval(m+'(denseit(final_test_vec_attr['+str(i)+',:]),denseit(final_test_stvec_attr['+str(i)+',:]))')
                distances_test[m+'_tf_attr'][i] = eval(m+'(denseit(final_test_vec_tf_attr['+str(i)+',:]),denseit(final_test_stvec_tf_attr['+str(i)+',:]))')

    ## create new train data with distance measurements
    logging.warn('Save distance data')
    train_full_distances = pd.DataFrame(distances_train)
    final_test_distances = pd.DataFrame(distances_test)
    train_full_distances.index = train_full.index
    final_test_distances.index = final_test.index
    train_full_distances.to_csv('Data/train_distances.csv', index=True)
    final_test_distances.to_csv('Data/test_distances.csv', index=True)

    ## add matching words count
    logging.warn('Calculate matching words and TF scores')
    # matching words overall
    matching_words_train, matching_words_test = np.zeros(N), np.zeros(N2)
    tf_max_score_train, tf_max_score_test = np.zeros(N), np.zeros(N2)
    tf_score_total_train, tf_score_total_test = np.zeros(N), np.zeros(N2)
    # matching words title
    matching_words_title_train, matching_words_title_test = np.zeros(N), np.zeros(N2)
    tf_max_score_title_train, tf_max_score_title_test = np.zeros(N), np.zeros(N2)
    tf_score_total_title_train, tf_score_total_title_test = np.zeros(N), np.zeros(N2)
    # matching words description
    matching_words_desc_train, matching_words_desc_test = np.zeros(N), np.zeros(N2)
    tf_max_score_desc_train, tf_max_score_desc_test = np.zeros(N), np.zeros(N2)
    tf_score_total_desc_train, tf_score_total_desc_test = np.zeros(N), np.zeros(N2)
    # matching words attributes
    matching_words_attr_train, matching_words_attr_test = np.zeros(N), np.zeros(N2)
    tf_max_score_attr_train, tf_max_score_attr_test = np.zeros(N), np.zeros(N2)
    tf_score_total_attr_train, tf_score_total_attr_test = np.zeros(N), np.zeros(N2)

    for i in range(max(N,N2)):
        if i<N:
            # overall
            matching_words_train[i] = np.sum(train_full_stvec[i,:].todense())
            tf_max_score_train[i] = np.max(train_full_stvec_tf[i,:].todense())
            tf_score_total_train[i] = np.sum(train_full_stvec_tf[i,:].todense())
            # title
            matching_words_title_train[i] = np.sum(train_full_stvec_title[i,:].todense())
            tf_max_score_title_train[i] = np.max(train_full_stvec_tf_title[i,:].todense())
            tf_score_total_title_train[i] = np.sum(train_full_stvec_tf_title[i,:].todense())
            # description
            matching_words_desc_train[i] = np.sum(train_full_stvec_desc[i,:].todense())
            tf_max_score_desc_train[i] = np.max(train_full_stvec_tf_desc[i,:].todense())
            tf_score_total_desc_train[i] = np.sum(train_full_stvec_tf_desc[i,:].todense())
            # attributes
            matching_words_attr_train[i] = np.sum(train_full_stvec_attr[i,:].todense())
            tf_max_score_attr_train[i] = np.max(train_full_stvec_tf_attr[i,:].todense())
            tf_score_total_attr_train[i] = np.sum(train_full_stvec_tf_attr[i,:].todense())
        if i<N2:
            # overall
            matching_words_test[i] = np.sum(final_test_stvec[i,:].todense())
            tf_max_score_test[i] = np.max(final_test_stvec_tf[i,:].todense())
            tf_score_total_test[i] = np.sum(final_test_stvec_tf[i,:].todense())
            # title
            matching_words_title_test[i] = np.sum(final_test_stvec_title[i,:].todense())
            tf_max_score_title_test[i] = np.max(final_test_stvec_tf_title[i,:].todense())
            tf_score_total_title_test[i] = np.sum(final_test_stvec_tf_title[i,:].todense())
            # description
            matching_words_desc_test[i] = np.sum(final_test_stvec_desc[i,:].todense())
            tf_max_score_desc_test[i] = np.max(final_test_stvec_tf_desc[i,:].todense())
            tf_score_total_desc_test[i] = np.sum(final_test_stvec_tf_desc[i,:].todense())
            # attributes
            matching_words_attr_test[i] = np.sum(final_test_stvec_attr[i,:].todense())
            tf_max_score_attr_test[i] = np.max(final_test_stvec_tf_attr[i,:].todense())
            tf_score_total_attr_test[i] = np.sum(final_test_stvec_tf_attr[i,:].todense())

    matching_scores_train = pd.DataFrame({
                                            'matching_words':matching_words_train
                                             , 'tf_max_score':tf_max_score_train
                                             , 'tf_score_total':tf_max_score_train
                                             , 'matching_words_title':matching_words_title_train
                                             , 'tf_max_score_title':tf_max_score_title_train
                                             , 'tf_score_total_title':tf_max_score_title_train
                                             , 'matching_words_desc':matching_words_desc_train
                                             , 'tf_max_score_desc':tf_max_score_desc_train
                                             , 'tf_score_total_desc':tf_max_score_desc_train
                                             , 'matching_words_attr':matching_words_attr_train
                                             , 'tf_max_score_attr':tf_max_score_attr_train
                                             , 'tf_score_total_attr':tf_max_score_attr_train
                                          }
                                          , index = train_full.index)
    matching_scores_train['matching_words_pct'] = matching_scores_train['matching_words'] \
                                                    / countwords(train_full['search_term'])
    matching_scores_train['matching_words_title_pct'] = matching_scores_train['matching_words_title'] \
                                                    / countwords(train_full['search_term'])
    matching_scores_train['matching_words_desc_pct'] = matching_scores_train['matching_words_desc'] \
                                                    / countwords(train_full['search_term'])
    matching_scores_train['matching_words_attr_pct'] = matching_scores_train['matching_words_attr'] \
                                                    / countwords(train_full['search_term'])
    matching_scores_test = pd.DataFrame({
                                            'matching_words':matching_words_test
                                             , 'tf_max_score':tf_max_score_test
                                             , 'tf_score_total':tf_max_score_test
                                             , 'matching_words_title':matching_words_title_test
                                             , 'tf_max_score_title':tf_max_score_title_test
                                             , 'tf_score_total_title':tf_max_score_title_test
                                             , 'matching_words_desc':matching_words_desc_test
                                             , 'tf_max_score_desc':tf_max_score_desc_test
                                             , 'tf_score_total_desc':tf_max_score_desc_test
                                             , 'matching_words_attr':matching_words_attr_test
                                             , 'tf_max_score_attr':tf_max_score_attr_test
                                             , 'tf_score_total_attr':tf_max_score_attr_test
                                         }
                                         , index = final_test.index)
    matching_scores_test['matching_words_pct'] = matching_scores_test['matching_words'] \
                                                    / countwords(final_test['search_term'])
    matching_scores_test['matching_words_pct'] = matching_scores_test['matching_words'] \
                                                    / countwords(final_test['search_term'])
    matching_scores_test['matching_words_title_pct'] = matching_scores_test['matching_words_title'] \
                                                    / countwords(final_test['search_term'])
    matching_scores_test['matching_words_desc_pct'] = matching_scores_test['matching_words_desc'] \
                                                    / countwords(final_test['search_term'])
    matching_scores_test['matching_words_attr_pct'] = matching_scores_test['matching_words_attr'] \
                                                    / countwords(final_test['search_term'])

    ## calculate further features
    P = 100

    ## Decompose features into smaller subset
    logging.warn('SVD of smaller search-term count vectors')
    svd = TruncatedSVD(P)
    cv = CountVectorizer(stop_words='english')
    tf = TfidfTransformer()

    train_full_stvec = cv.fit_transform(train_full['search_term_original'])
    final_test_stvec = cv.transform(final_test['search_term_original'])
    train_full_stvec_tf = tf.fit_transform(train_full_stvec)
    final_test_stvec_tf = tf.transform(final_test_stvec)
    matching_scores_train['matching_words_pct_original'] = np.sum(train_full_stvec,axis=1) \
                                                            / np.sum(cv.transform(train_full['product_title'] \
                                                                                    +' '+train_full['description_words'] \
                                                                                    +' '+train_full['attribute_words']),axis=1)
    matching_scores_test['matching_words_pct_original'] = np.sum(final_test_stvec,axis=1) \
                                                            / np.sum(cv.transform(final_test['product_title'] \
                                                                                    +' '+final_test['description_words'] \
                                                                                    +' '+final_test['attribute_words']),axis=1)
    train_full_vec_tf = tf.fit_transform(train_full_vec)
    final_test_vec_tf = tf.transform(final_test_vec)
    train_search_term_svd = svd.fit_transform(train_full_stvec)
    train_search_term_svd = pd.DataFrame(train_search_term_svd,columns=['svd_'+str(i) for i in range(P)])
    train_search_term_svd.index = train_full.index
    final_search_term_svd = svd.transform(final_test_stvec)
    final_search_term_svd = pd.DataFrame(final_search_term_svd,columns=['svd_'+str(i) for i in range(P)])
    final_search_term_svd.index = final_test.index

    # combined all datasets
    logging.warn('Combining datasets')
    train_full = pd.concat((train_full_distances
                            , matching_scores_train
                            , train_search_term_km
                            , train_search_term_svd ), axis=1)
    final_test = pd.concat((final_test_distances
                            , matching_scores_test
                            , final_test_km
                            , final_search_term_svd ), axis=1)

    # save new training and test datasets
    logging.warn('Save datasets to disk')
    train_full.to_csv('Data/train_full_v2.csv',index=True)
    final_test.to_csv('Data/final_test_v2.csv',index=True)

# ### Run forest model ##
def forest_model(save_final_results=True, test=True):
    ''' execute final model
    '''
    global train_full
    global target_full
    global f_pred
    global PARAMS

    logging.warn('Create boosted trees model with training data')
    ## Encode categories ##
    im = Imputer(strategy='median')
    X = im.fit_transform(np.matrix(train_full))
    Y = np.array(target_full[TARGET_COLUMN]).ravel()
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( \
                                                      X \
                                                      , Y \
                                                      , test_size=TEST_SIZE \
                                                      , random_state=0)
    rfr = RandomForestRegressor(**PARAMS)
    if test:
        logging.warn('Test prediction accuracy')
        rfr.fit(X_train , Y_train)
        p_pred = rfr.predict(X_test)
        logging.warn('RMSE: '+str(np.sqrt(np.mean((p_pred-Y_test)**2))))

    ## Run model with all data and save ##
    if save_final_results:
        ''' Write results to a csv file
            NOTE: sorting is not done here
        '''
        rfr.fit(X , Y)
        X = im.transform(np.matrix(final_test))
        f_pred = rfr.predict(X)

def write_submission():
    global f_pred
    global final_test
    ## Write to submission file ##
    preds = f_pred
    ids = final_test.index.ravel()
    results_df = pd.DataFrame({'id':ids,'relevance':preds})
    results_df.to_csv('Data/submission.csv',index=False)

def run():
    global train_full
    global target_full
    global final_test
    global f_pred
    global accuracies
    global GS_CV

    # Load data and declare arguments
    declare_args(); load_data()

    # load description data
    load_descriptions(refresh=False)

    # load attributes data
    load_attributes(refresh=False)

    # combine datasets to final
    combine_data()

    # run forest model
    forest_model(test=True, save_final_results=True)

    # Write submission file
    write_submission()

# if __name__=='__main__':
#     run()
