
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
import sys
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
synonym_text = np.vectorize(lambda x: ' '.join(' '.join(synonyms(z) if synonyms(z) else []) for z in x.split()))

# custom functions
categorize = np.vectorize(lambda x: round(x,1))
WORDS = re.compile(r'[a-zA-Z]+')
findwords = np.vectorize(lambda x: ' '.join(WORDS.findall(str(x).lower())))

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
    global GS_CV

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

    # XGA boost params
    GS_CV = {'subsample': 0.5, 'colsample_bytree': 0.5, 'max_depth': 8}

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

def load_attributes():
    global attributes
    logging.warn('Compiling attribute dimensions per product')

    ## create matrix of attributes by product
    attributes['words'] = findwords(attributes['value'])
    attributes['words'] = attributes['words'] + ' '
    attributes_new = pd.DataFrame(attributes['words'].groupby(level=0).sum())
    attributes_new.columns = ['attribute_words']
    attributes_new.index = pd.Int64Index(attributes_new.index)
    attributes = attributes_new
    del attributes_new

def load_descriptions():
    global descriptions
    logging.warn('Compiling descriptions into features for each product')

    ## filter out non-words
    descriptions['description_words'] = findwords(descriptions['product_description'])
    descriptions_new = pd.DataFrame(descriptions['description_words'],index=descriptions['product_uid'])
    descriptions = descriptions_new
    del descriptions_new

def combine_data():
    global train_full
    global final_test
    logging.warn('Combining all datasets')

    ## filter out non-words in title
    train_full['product_title'] = findwords(train_full['product_title'])
    final_test['product_title'] = findwords(final_test['product_title'])

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

    ## Set ID's back
    train_full.index = train_full['id']
    final_test.index = final_test['id']

    ## vectorize words in attributes
    logging.warn('Calculate count and TF-IDF vectors for title+description+attributes')
    cv = CountVectorizer(stop_words='english')
    tf = TfidfTransformer()
    train_full_vec = cv.fit_transform(train_full['product_title'].fillna('')
                                    +' '+train_full['description_words'].fillna('')
                                    +' '+train_full['attribute_words'].fillna(''))
    final_test_vec = cv.transform(final_test['product_title'].fillna('')
                                    +' '+final_test['description_words'].fillna('')
                                    +' '+final_test['attribute_words'].fillna(''))
    train_full_vec_tf = tf.fit_transform(train_full_vec)
    final_test_vec_tf = tf.transform(final_test_vec)

    ## vectorize search terms
    logging.warn('Calculate count and TF-IDF vectors for search terms')
    # train_full['search_term_expanded'] = train_full['search_term']+' '+synonym_text(train_full['search_term'])
    # final_test['search_term_expanded'] = final_test['search_term']+' '+synonym_text(final_test['search_term'])
    train_full_stvec = cv.transform(train_full['search_term'].apply(lambda x: x.lower()))
    final_test_stvec = cv.transform(final_test['search_term'].apply(lambda x: x.lower()))
    train_full_stvec_tf = tf.transform(train_full_stvec)
    final_test_stvec_tf = tf.transform(final_test_stvec)

    ## calculate distances between search terms and product data
    logging.warn('Run distance calculations')
    denseit = lambda x: np.array(x.todense()).ravel()
    N = train_full_vec.shape[0]
    N2 = final_test_vec.shape[0]
    metrics = ['euclidean','braycurtis'
                ,'canberra','correlation','cityblock'
                ,'hamming','chebyshev','cosine','dice','rogerstanimoto'
                ,'sokalmichener','sokalsneath'
                ,'sqeuclidean']
    distances_train = {}
    distances_test = {}
    for m in metrics:
        distances_train[m] = np.zeros(N)
        distances_train[m+'_tf'] = np.zeros(N)
        distances_test[m] = np.zeros(N2)
        distances_test[m+'_tf'] = np.zeros(N2)
    for m in metrics:
        logging.warn('Calculating distance metric {}'.format(m))
        for i in range(max(N,N2)):
            if i<N:
                distances_train[m][i] = eval(m+'(denseit(train_full_vec['+str(i)+',:]),denseit(train_full_stvec['+str(i)+',:]))')
                distances_train[m+'_tf'][i] = eval(m+'(denseit(train_full_vec_tf['+str(i)+',:]),denseit(train_full_stvec_tf['+str(i)+',:]))')
            if i<N2:
                distances_test[m][i] = eval(m+'(denseit(final_test_vec['+str(i)+',:]),denseit(final_test_stvec['+str(i)+',:]))')
                distances_test[m+'_tf'][i] = eval(m+'(denseit(final_test_vec_tf['+str(i)+',:]),denseit(final_test_stvec_tf['+str(i)+',:]))')

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
    matching_words_train = np.zeros(N)
    tf_max_score_train = np.zeros(N)
    tf_score_total_train = np.zeros(N)
    matching_words_test = np.zeros(N2)
    tf_max_score_test = np.zeros(N2)
    tf_score_total_test = np.zeros(N2)
    for i in range(max(N,N2)):
        if i<N:
            matching_words_train[i] = np.sum(train_full_stvec[i,:].todense())
            tf_max_score_train[i] = np.max(train_full_stvec_tf[i,:].todense())
            tf_score_total_train[i] = np.sum(train_full_stvec_tf[i,:].todense())
        if i<N2:
            matching_words_test[i] = np.sum(final_test_stvec[i,:].todense())
            tf_max_score_test[i] = np.max(final_test_stvec_tf[i,:].todense())
            tf_score_total_test[i] = np.sum(final_test_stvec_tf[i,:].todense())

    ## calculate further features
    K = 6
    P = 100

    ## KMeans clustering
    logging.warn('KMeans clustering')
    km = KMeans(K)
    train_search_term_km = km.fit_predict(train_full_stvec)
    train_search_term_km = pd.DataFrame({'km':train_search_term_km})
    train_search_term_km.index = train_full.index
    final_test_km = km.predict(final_test_stvec)
    final_test_km = pd.DataFrame({'km':final_test_km})
    final_test_km.index = final_test.index

    ## Decompose features into smaller subset
    logging.warn('SVD of search-term count vectors')
    svd = TruncatedSVD(P)
    train_search_term_svd = svd.fit_transform(train_full_stvec)
    train_search_term_svd = pd.DataFrame(train_search_term_svd,columns=['svd_'+str(i) for i in range(P)])
    train_search_term_svd.index = train_full.index
    final_search_term_svd = svd.transform(final_test_stvec)
    final_search_term_svd = pd.DataFrame(final_search_term_svd,columns=['svd_'+str(i) for i in range(P)])
    final_search_term_svd.index = final_test.index

    # combined all datasets
    logging.warn('Combining datasets')
    train_full = pd.concat((train_full_distances
                            , matching_words_train
                            , tf_max_score_train
                            , tf_score_total_train
                            , train_search_term_km
                            , train_search_term_svd ), axis=1)
    final_test = pd.concat((final_test_distances
                            , matching_words_test
                            , tf_max_score_test
                            , tf_score_total_test
                            , final_test_km
                            , final_search_term_svd ), axis=1)

# ### Run forest model ##
def forest_model(test=True,grid_cv=False,save_final_results=True):
    ''' execute final model
    '''
    global train_full
    global target_full
    global TRAIN_COLUMNS
    global GS_CV
    global f_pred
    global accuracies

    logging.warn('Create boosted trees model with training data')
    ## Encode categories ##
    le = LabelEncoder()
    X = np.matrix(train_full.fillna(0))
    Y = le.fit_transform(categorize(np.array(target_full[TARGET_COLUMN]).ravel()))
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( \
                                                      X \
                                                      , Y \
                                                      , test_size=TEST_SIZE \
                                                      , random_state=0)

    if grid_cv:
        ## Run grid search to find optimal parameters ##
        params_grid = {
        		         'max_depth':[ 15, 20, 25 ] ,
                         'subsample':[ 0.25, 0.5  ] ,
                        #  'colsample_bytree':[ 0.25, 0.5, 0.75 ] ,
                }
        logging.warn('Running grid search CV with params: {}'.format(params_grid))
        xgb = XGBClassifier(n_estimators=50, objective='multi:softprob', seed=0)
        cv = GridSearchCV(xgb, params_grid).fit(X, Y)
        logging.warn('Best XGB params: {}'.format(cv.best_params_))
        GS_CV = cv.best_params_

    ## Run model with all data and save ##
    if save_final_results:
        ''' Write results to a csv file
            NOTE: sorting is not done here
        '''
        logging.warn('Make predictions for final test set with settings: {}'.format(GS_CV))
        xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000,
                            objective='multi:softprob',seed=0, **GS_CV)
        xgb.fit(X_train , Y_train)
        if test:
            logging.warn('Test prediction accuracy')
            p_pred = xgb.predict(X_test)
            p_pred_p = xgb.predict_proba(X_test)
            logging.warn('Accuracy: '+str(np.mean(p_pred == Y_test)))
            logging.warn('\n'+classification_report(p_pred, Y_test))
            logging.warn('Log Loss: {}'.format(log_loss(Y_test, p_pred_p)))
            categories = set(Y_test-1)
            accuracies = np.zeros(len(categories))
            for c in categories:
                accuracies[c] = np.sum(p_pred[p_pred-1==c]==Y_test[p_pred-1==c])*1.0
                accuracies[c] /= p_pred[p_pred-1==c].shape[0]

        X = np.matrix(final_test.fillna(0))
        f_pred = xgb.predict_proba(X)

def compile_nn(input_dim, output_dim, size=512):
    ## Layer 1
    size = size
    nn_model = Sequential()
    nn_model.add(Dense(input_dim, size, init='glorot_uniform', W_regularizer=l2(0.1)))
    nn_model.add(PReLU(size))
    nn_model.add(Dropout(0.5))

    ## Layer 2
    nn_model.add(Dense(size, size))
    nn_model.add(PReLU(size))
    nn_model.add(Dropout(0.5))

    ## Layer 3
    nn_model.add(Dense(size, size))
    nn_model.add(PReLU(size))
    nn_model.add(Dropout(0.5))

    nn_model.add(Dense(input_dim=size, output_dim=output_dim))
    nn_model.add(Activation("softmax"))
    nn_model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return nn_model

def neural_model(test=True,save_final_results=True):
    global train_full
    global target_full
    global accuracies
    global TRAIN_COLUMNS

    logging.warn('Create neural model with training data')

    ## Set up X,Y data for modeling ##
    lb = LabelBinarizer()
    le = LabelEncoder()
    X = np.matrix(train_full.fillna(0))
    c = le.fit_transform(categorize(np.array(target_full[TARGET_COLUMN]).ravel()))
    Y = lb.fit_transform(c)
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( \
                                                      X \
                                                      , Y \
                                                      , test_size=TEST_SIZE \
                                                      , random_state=0)

    ## Neural Network ##
    model = compile_nn(X.shape[1],Y.shape[1], 1024)

    if save_final_results:
        ''' Write results to a csv file
            NOTE: sorting is not done here
        '''
        logging.warn('Make predictions for final test set')
        model.fit(X_train, Y_train, nb_epoch=25, batch_size=128)

        if test:
            logging.warn('Test prediction accuracy')
            p_pred = model.predict(X_test)
            p_pred_i = lb.inverse_transform(p_pred)
            Y_test_i = lb.inverse_transform(Y_test)

            logging.warn('Accuracy: '+str(np.mean(p_pred_i == Y_test_i)))
            logging.warn('\n'+classification_report(p_pred_i, Y_test_i))
            logging.warn('Log Loss: {}'.format(log_loss(Y_test, p_pred_p)))

            categories = set(Y_test_i-1)
            accuracies = np.zeros(len(categories))
            for c in categories:
                accuracies[c] = np.sum(p_pred_i[p_pred_i-1==c]==Y_test_i[p_pred_i-1==c])*1.0
                accuracies[c] /= p_pred_i[p_pred_i-1==c].shape[0]

        X = np.matrix(final_test.fillna(0))
        f_pred = model.predict_proba(X)

def logmodels(test=True,save_final_results=True,n_k=4):
    global train_full
    global target_full
    global X_train
    global X_test
    global Y_train
    global Y_test
    global final_X_test
    global GS_CV
    global f_pred
    global accuracies

    logging.warn('Create iterative clustered logistic regression model')
    ## Encode categories ##
    lb = LabelBinarizer()
    le = LabelEncoder()
    X = np.matrix(train_full.fillna(0))
    X_final = np.matrix(final_test.fillna(0))
    c = le.fit_transform(categorize(np.array(target_full[TARGET_COLUMN]).ravel()))
    Y = lb.fit_transform(c)
    categories = range(Y.shape[1])

    ## Initialize Clusters ##
    km = KMeans(n_clusters=n_k)
    clusters = km.fit_predict(X)
    clusters_p = km.predict(X_final)
    X = np.concatenate((np.matrix(clusters).T,X),axis=1)
    X_final = np.concatenate((np.matrix(clusters_p).T,X_final),axis=1)
    cluster_set = range(n_k)

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( \
                                                      X \
                                                      , Y \
                                                      , test_size=TEST_SIZE \
                                                      , random_state=0)

    if save_final_results:
        ''' Write results to a csv file
            NOTE: sorting is not done here
        '''
        logging.warn('Make predictions for final test set')
        cluster_models = {}
        f_pred = np.zeros((X_final.shape[0],Y_train.shape[1]))
        p_pred = np.zeros((X_test.shape[0],Y_test.shape[1]))
        for k in cluster_set:
            logging.warn('Running logistic model for {} cluster'.format(k))
            for c in categories:
                # find cluster subset
                k_locs = np.array(X_train[:,0]==k).ravel()
                k_test_locs = np.array(X_test[:,0]==k).ravel()
                kp_locs = np.array(X_final[:,0]==k).ravel()

                # subset data
                X_k = X_train[k_locs,:]
                X_k_final = X_final[kp_locs,:]
                Y_k = Y_train[k_locs,c]
                X_k_test = X_test[k_test_locs,:]

                # initialize model for subset
                lm = LogisticRegression()
                lm.fit(X_k, Y_k)

                # predict probability
                f_pred_tmp = lm.predict_proba(X_k_final)
                f_pred_test_tmp = lm.predict_proba(X_k_test)

                # fill subset predicted value
                f_pred[kp_locs,c] = f_pred_tmp[:,1]
                p_pred[k_test_locs,c] = f_pred_test_tmp[:,1]
                if k not in cluster_models:
                    cluster_models[k] = []
                cluster_models[k].append(lm)

        if test:
            logging.warn('Test prediction accuracy')
            p_pred_i = np.argmax(p_pred, axis=1)+1
            p_pred_lb = lb.transform(p_pred_i)
            Y_test = lb.inverse_transform(Y_test)
            p_pred_p = p_pred
            logging.warn('Accuracy: {}'.format(str(np.mean(p_pred_i == Y_test))))
            logging.warn('\n'+classification_report(p_pred_i, Y_test))
            logging.warn('Log Loss: {}'.format(log_loss(Y_test, p_pred_p)))

            categories = set(Y_test-1)
            accuracies = np.zeros(len(categories))
            for c in categories:
                accuracies[c] = np.sum(p_pred_i[p_pred_i-1==c]==Y_test[p_pred_i-1==c])*1.0
                accuracies[c] /= p_pred_i[p_pred_i-1==c].shape[0]

def write_submission():
    global f_pred
    global final_test
    ## Write to submission file ##
    roundpreds = np.vectorize(lambda x: int(round(x,0)))
    preds = roundpreds(np.argmax(f_pred,axis=1)+1)
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
    load_descriptions()

    # load attributes data
    load_attributes()

    # combine datasets to final
    combine_data()

    # Run forest model
    GS_CV = {'subsample': 0.5, 'colsample_bytree': 0.5, 'max_depth': 12}
    forest_model(test=True, grid_cv=False, save_final_results=True)
    f_pred_for = f_pred
    accuracies_for = accuracies

    # Run neural model
    neural_model(test=True, save_final_results=True)
    f_pred_nn = f_pred
    accuracies_nn = accuracies

    # Run clustered LR models
    logmodels(test=True, save_final_results=True, n_k=3)
    f_pred_log = f_pred
    accuracies_log = accuracies

    # Combine models
    accuracies_for[np.isnan(accuracies_for)] = 0
    accuracies_log[np.isnan(accuracies_log)] = 0
    w_for = accuracies_for/(accuracies_for+accuracies_log)
    w_for[np.isnan(w_for)] = 0
    w_log = 1-w_for
    f_pred = f_pred_log*w_log + f_pred_for*w_for

    # Write submission file
    write_submission()

# if __name__=='__main__':
#     run()
