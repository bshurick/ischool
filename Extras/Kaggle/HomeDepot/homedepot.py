
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
from sklearn.decomposition import PCA
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

# custom MCL function
class MultiColumnLabelEncoder:
    ''' Create a class that encodes
        labels for a matrix of data
    '''
    def __init__(self, columns = None):
        self.columns = columns # array of column names to encode
        self.encoders = {}

    def fit(self,X,y=None):
        return self # not relevant here

    def get_params(self, deep=True):
        out = dict()
        if self.columns: out['columns'] = columns
        return out

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). NOTE: Assumes use of Pandas DataFrame
        '''
        numerics = [np.float16, np.float32, np.float64]
        ints = [np.int16, np.int32, np.int64]
        output = X.copy()

        for colname,col in output.iteritems():
            if col.dtype not in numerics+ints:
                le = LabelEncoder()
                output[colname] = le.fit_transform(output[colname])
                self.encoders[colname] = le
            elif col.dtype in numerics+ints:
                pass
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

    def inverse_transform(self, X, y=None):
        '''
        Inverse of transform function
        NOTE: still assumes use of pandas DataFrame
        '''
        numerics = [np.float16, np.float32, np.float64]
        ints = [np.int16, np.int32, np.int64]
        output = X.copy()

        for colname in self.encoders:
            le = self.encoders[colname]
            output[colname] = le.inverse_transform(output[colname])
        return output

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
    global TRAIN_COLUMNS
    logging.warn('Compiling attribute dimensions per product')

    ## create matrix of attributes by product
    WORDS = re.compile(r'[a-zA-Z]+')
    findwords = np.vectorize(lambda x: ' '.join(WORDS.findall(str(x))))
    attributes['words'] = findwords(attributes['value'])

    ## vectorize words in attributes
    cv = CountVectorizer(stop_words='english', max_features=1000, max_df=0.25)
    attribute_vec = cv.fit_transform(attributes['words'])

    ## Function to minimize session data at the user level ##
    def minimize_df(i,lim,n,nparr):
        m = min(lim,(i+1)*n)
        return pd.DataFrame(nparr[:,i*n:m].toarray() \
                    ,index=attributes['product_uid']) \
                    .groupby(level=0).sum()

    ## Minimize word counts data ##
    n = attribute_vec.shape[1]//50 # 50 chunks of word count data
    z = ( minimize_df(y,attribute_vec.shape[1],50,attribute_vec) for y in range(n+1) )
    attributes_new = pd.concat(z,axis=1)
    ## add Tf-Idf transformer here!
    attributes_new.columns = [ 'attribute_'+str(i) for i in range(len(attributes_new.columns)) ]
    TRAIN_COLUMNS += list(attributes_new.columns)
    attributes = attributes_new
    del attributes_new

def load_descriptions():
    global descriptions
    global TRAIN_COLUMNS
    logging.warn('Compiling descriptions into features for each product')

    ## filter out non-words
    WORDS = re.compile(r'[a-zA-Z]+')
    findwords = np.vectorize(lambda x: ' '.join(WORDS.findall(str(x))))
    descriptions['words'] = findwords(descriptions['product_description'])

    ## vectorize words in attributes
    ## add Tf-Idf vectorizer here!
    cv = CountVectorizer(stop_words='english',max_features=1000, max_df=0.25)
    descriptions_vec = cv.fit_transform(descriptions['words'])
    descriptions_new = pd.DataFrame(descriptions_vec.todense(),index=descriptions['product_uid'])
    descriptions_new.columns = [ 'description_'+str(i) for i in range(len(descriptions_new.columns)) ]
    TRAIN_COLUMNS += list(descriptions_new.columns)
    descriptions = descriptions_new
    del descriptions_new

def manipulate_train_data():
    global train_full
    global final_test
    global TRAIN_COLUMNS
    logging.warn('Compiling descriptions into features for each product')

    ## filter out non-words in title
    WORDS = re.compile(r'[a-zA-Z]+')
    findwords = np.vectorize(lambda x: ' '.join(WORDS.findall(str(x))))
    train_full['product_title'] = findwords(train_full['product_title'])
    final_test['product_title'] = findwords(final_test['product_title'])

    ## vectorize words in attributes
    ## add Tf-Idf transformer here!
    cv = CountVectorizer(stop_words='english',max_features=1000, max_df=0.25)
    train_full_vec = cv.fit_transform(train_full['product_title'])
    final_test_vec = cv.transform(final_test['product_title'])

    ## combine data with product index
    train_full_new = pd.DataFrame(train_full_vec.todense(),index=train_full['id'])
    train_full_new.columns = [ 'title_'+str(i) for i in range(len(train_full_new.columns)) ]
    final_test_new = pd.DataFrame(final_test_vec.todense(),index=final_test['id'])
    final_test_new.columns = [ 'title_'+str(i) for i in range(len(final_test_new.columns)) ]
    TRAIN_COLUMNS += list(train_full_new.columns)

    ## vectorize search terms
    cv2 = CountVectorizer(stop_words='english',max_features=1000, max_df=0.25)
    train_full_stvec = cv2.fit_transform(train_full['search_term'])
    final_test_stvec = cv2.transform(final_test['search_term'])

    ## combine search term data
    train_full_stnew = pd.DataFrame(train_full_stvec.todense(),index=train_full['id'])
    train_full_stnew.columns = [ 'search_'+str(i) for i in range(len(train_full_stnew.columns)) ]
    final_test_stnew = pd.DataFrame(final_test_stvec.todense(),index=final_test['id'])
    final_test_stnew.columns = [ 'search_'+str(i) for i in range(len(final_test_stnew.columns)) ]
    train_full_new = pd.concat((train_full, train_full_new, train_full_stnew),axis=1)
    final_test_new = pd.concat((final_test, final_test_new, final_test_stnew),axis=1)
    TRAIN_COLUMNS += list(train_full_stnew.columns)
    train_full = train_full_new
    final_test = final_test_new
    del train_full_new
    del final_test_new

def combine_data():
    global descriptions
    global attributes
    global train_full
    global target_full
    global final_test
    logging.warn('Combining datasets')

    ## attach product data to training set
    train_full.index = train_full['product_uid']
    final_test.index = final_test['product_uid']

    ## merge attribute data
    train_full = pd.merge(train_full, attributes, how='left', left_index=True, right_index=True)
    final_test = pd.merge(final_test, attributes, how='left', left_index=True, right_index=True)

    ## merge description data
    train_full = pd.merge(train_full, descriptions, how='left', left_index=True, right_index=True)
    final_test = pd.merge(final_test, descriptions, how='left', left_index=True, right_index=True)

    ## reset indices
    train_full.index = train_full['id']
    final_test.index = final_test['id']

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
    categorize = np.vectorize(lambda x: int(round(x,0)))

    logging.warn('Create boosted trees model with training data')
    ## Encode categories ##
    X = np.matrix(train_full[TRAIN_COLUMNS].fillna(0))
    Y = categorize(np.array(target_full[TARGET_COLUMN]).ravel())
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
        logging.warn('Make predictions for final test set')
        xgb = XGBClassifier(learning_rate=0.1, n_estimators=250,
                            objective='multi:softprob',seed=0, **GS_CV)
        xgb.fit(X_train , Y_train)
        if test:
            logging.warn('Test prediction accuracy')
            p_pred = xgb.predict(X_test)
            p_pred_p = xgb.predict_proba(X_test)
            cat_tst_lb = lb.fit_transform(Y_test)
            logging.warn('Accuracy: '+str(np.mean(p_pred == Y_test)))
            logging.warn('\n'+classification_report(p_pred, Y_test))
            logging.warn('Log Loss: {}'.format(log_loss(Y_test, p_pred_p)))
            logging.warn('Label Ranking Precision score: {}'\
                            .format(label_ranking_average_precision_score(cat_tst_lb, p_pred_p)))
            logging.warn('Label Ranking loss: {}'.format(label_ranking_loss(cat_tst_lb, p_pred_p)))
            categories = set(Y_test)
            accuracies = np.zeros(len(categories))
            for c in categories:
                accuracies[c] = np.sum(p_pred[p_pred==c]==Y_test[p_pred==c])*1.0
                accuracies[c] /= p_pred[p_pred==c].shape[0]

        X = np.matrix(final_test[TRAIN_COLUMNS].fillna(0))
        f_pred = xgb.predict_proba(X)

def compile_nn(input_dim, output_dim):
    ## Layer 1
    nn_model = Sequential()
    nn_model.add(Dense(output_dim=1024, input_dim=(input_dim,), W_regularizer=l2(0.1)))
    nn_model.add(PReLU())
    nn_model.add(BatchNormalization())
    nn_model.add(Dropout(0.5))

    ## Layer 2
    nn_model.add(Dense(1024))
    nn_model.add(PReLU())
    nn_model.add(BatchNormalization())
    nn_model.add(Dropout(0.5))

    ## Layer 3
    nn_model.add(Dense(1024))
    nn_model.add(PReLU())
    nn_model.add(BatchNormalization())
    nn_model.add(Dropout(0.5))

    nn_model.add(Dense(output_dim=output_dim))
    nn_model.add(Activation("softmax"))
    nn_model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return nn_model

def neural_model(test=True,save_final_results=True):
    global train_full
    global target_full
    global TRAIN_COLUMNS
    categorize = np.vectorize(lambda x: int(round(x,0)))

    logging.warn('Create neural model with training data')

    ## Set up X,Y data for modeling ##
    lb = LabelBinarizer()
    X = np.matrix(train_full[TRAIN_COLUMNS].fillna(0))
    Y = lb.fit_transform(categorize(np.array(target_full[TARGET_COLUMN]).ravel()))
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( \
                                                      X \
                                                      , Y \
                                                      , test_size=TEST_SIZE \
                                                      , random_state=0)

    ## Neural Network ##
    model = compile_nn(X.shape[1],Y.shape[1])

    if save_final_results:
        ''' Write results to a csv file
            NOTE: sorting is not done here
        '''
        logging.warn('Make predictions for final test set')
        model.fit(X_train, Y_train, nb_epoch=10, batch_size=128)

        if test:
            logging.warn('Test prediction accuracy')
            p_pred = model.predict(X_test)
            p_pred_i = lb.inverse_transform(p_pred)

            logging.warn('Accuracy: '+str(np.mean(p_pred_i == lb.inverse_transform(Y_test))))
            logging.warn('\n'+classification_report(p_pred_i, Y_test))
            logging.warn('Log Loss: {}'.format(log_loss(cat_tst_lb, p_pred_p)))
            logging.warn('Label Ranking Precision score: {}'\
                            .format(label_ranking_average_precision_score(cat_tst_lb, p_pred_p)))
            logging.warn('Label Ranking loss: {}'.format(label_ranking_loss(cat_tst_lb, p_pred_p)))

            cat_tst_le = le.transform(Y_test)
            p_pred_le = le.transform(p_pred_i)
            categories = set(cat_tst_le)
            accuracies = np.zeros(len(categories))
            for c in categories:
                accuracies[c] = np.sum(p_pred_le[p_pred_le==c]==cat_tst_le[p_pred_le==c])*1.0
                accuracies[c] /= p_pred_le[p_pred_le==c].shape[0]

        X = np.concatenate((p.transform(final_X_test[CAT_COLS]).todense() \
                                ,im2.transform(np.array(final_X_test[NUM_COLS]))),axis=1)
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
    le = LabelEncoder()
    lb = LabelBinarizer()
    cat_full = le.fit_transform(np.array(target_full).ravel())
    cat_full_lb = lb.fit_transform(np.array(target_full).ravel())

    mcl = MultiColumnLabelEncoder() ; ohe = OneHotEncoder() ; im = Imputer(strategy='most_frequent')
    im2 = Imputer(strategy='mean')
    p = Pipeline([('mcl',mcl),('im',im),('ohe',ohe)])

    ## full dataset ##
    X = np.concatenate((p.fit_transform(train_full[CAT_COLS]).todense() \
                            ,im2.fit_transform(np.array(train_full[NUM_COLS]))),axis=1)
    X_final = np.concatenate((p.transform(final_X_test[CAT_COLS]).todense() \
                            ,im2.transform(np.array(final_X_test[NUM_COLS]))),axis=1)
    Y = cat_full_lb
    categories = set(cat_full)

    ## Initialize Clusters ##
    km = KMeans(n_clusters=n_k)
    clusters = km.fit_predict(X)
    clusters_p = km.predict(X_final)
    X = np.concatenate((np.matrix(clusters).T,X),axis=1)
    X_final = np.concatenate((np.matrix(clusters_p).T,X_final),axis=1)
    cluster_set = set(clusters.tolist())

    ## Set up X,Y data for modeling ##
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
            p_pred_i = np.argmax(p_pred,axis=1)
            cat_tst_lb = Y_test
            Y_test = le.transform(lb.inverse_transform(Y_test))
            p_pred_lb = lb.transform(p_pred_i)
            p_pred_p = p_pred
            logging.warn('Accuracy: '+str(np.mean(p_pred_i == Y_test)))
            logging.warn('\n'+classification_report(p_pred_i, Y_test))
            logging.warn('Log Loss: {}'.format(log_loss(Y_test, p_pred_p)))
            logging.warn('Label Ranking Precision score: {}'\
                            .format(label_ranking_average_precision_score(cat_tst_lb, p_pred_p)))
            logging.warn('Label Ranking loss: {}'.format(label_ranking_loss(cat_tst_lb, p_pred_p)))
            logging.warn('NDCG score: {}'.format(ndcg_score(cat_tst_lb, p_pred_p, k=5)))

            categories = set(Y_test)
            accuracies = np.zeros(len(categories))
            for c in categories:
                accuracies[c] = np.sum(p_pred_i[p_pred_i==c]==Y_test[p_pred_i==c])*1.0
                accuracies[c] /= p_pred_i[p_pred_i==c].shape[0]

def write_submission():
    global f_pred
    global final_test
    ## Write to submission file ##
    roundpreds = np.vectorize(lambda x: int(round(x,0)))
    preds = roundpreds(np.argmax(f_pred,axis=1)+1)
    ids = final_test['id'].ravel()
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

    # load attribute data
    load_attributes()

    # manipulate_train_data
    manipulate_train_data()

    # combine datasets to final
    combine_data()

    # Run forest model
    GS_CV = {'subsample': 0.25, 'colsample_bytree': 0.25, 'max_depth': 6}
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
