
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
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, ElasticNetCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

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

def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """ adapted from https://gist.github.com/mblondel/7337391
    Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples, n_categories]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples, n_categories]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    if y_true is y_score:
        order = np.zeros((y_true.shape[0],k))
        order[:,0] = 1
        y_true = order
    else:
        if y_true.ndim == 1:
            lb = LabelBinarizer()
            y_true = lb.fit_transform(y_true)
        order = np.argsort(y_score)[:,::-1]
        y_true = np.take(y_true, order[:,:k])
    # y_score = np.sort(y_score)[:,::-1][:,:k]

    if gains == "exponential":
        gains = (2 ** y_true - 1)
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = 1/(np.log2(np.arange(y_true.shape[1]) + 2))

    return np.sum(gains * discounts)

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """ adapted from https://gist.github.com/mblondel/7337391
    Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples, n_categories]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples, n_categories]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

# ### Declare Args
def declare_args():
    ''' Declare global arguments and strings
    '''
    global AGE_GENDER_BUCKETS_FILE
    global COUNTRIES_FILE
    global SAMPLE_SUBMISSION_FILE
    global SESSIONS_FILE
    global TEST_DATA_FINAL_FILE
    global TRAIN_DATA_FILE
    global TEST_SIZE
    global USER_COLUMNS
    global TARGET_COLUMN
    global SESSION_COLUMNS
    global AGE_BUCKET_COLUMNS
    global CAT_COLS
    global NUM_COLS
    global GS_CV

    ## Files ##
    AGE_GENDER_BUCKETS_FILE = 'Data/age_gender_bkts.csv'
    COUNTRIES_FILE = 'Data/countries.csv'
    SAMPLE_SUBMISSION_FILE = 'Data/sample_submission.csv'
    SESSIONS_FILE = 'Data/sessions.csv'
    TEST_DATA_FINAL_FILE = 'Data/test_users.csv'
    TRAIN_DATA_FILE = 'Data/train_users_2.csv'

    ## Model args ##
    TEST_SIZE = 0.4

    ## Fields ##
    USER_COLUMNS = [
     'id',
     'date_account_created',
     'timestamp_first_active',
     'date_first_booking',
     'gender',
     'age',
     'signup_method',
     'signup_flow',
     'language',
     'affiliate_channel',
     'affiliate_provider',
     'first_affiliate_tracked',
     'signup_app',
     'first_device_type',
     'first_browser',
    ]
    TARGET_COLUMN = ['country_destination']

    SESSION_COLUMNS = [
     'user_id',
     'action',
     'action_type',
     'action_detail',
     'device_type',
     'secs_elapsed'
    ]

    AGE_BUCKET_COLUMNS = [
     'age_bucket',
     'country_destination',
     'gender',
     'population_in_thousands',
     'year'
    ]

    ## Define category and numeric fields for model ##
    CAT_COLS = [
     'gender',
     'signup_method',
     'signup_flow',
     'language',
     'affiliate_channel',
     'affiliate_provider',
     'first_affiliate_tracked',
     'signup_app',
     'first_device_type',
     'first_browser',
    ]
    NUM_COLS = [
        'age',
    ]

    # XGA boost params
    # GS_CV = {'subsample': 0.5, 'colsample_bytree': 0.5, 'max_depth': 8}
    GS_CV = {'subsample': 0.25, 'colsample_bytree': 0.75, 'max_depth': 12}

# ### Read data
def load_data():
    ''' read in data files
    '''
    global train_full
    global target_full
    global X_train
    global X_test
    global Y_train
    global Y_test
    global final_X_test
    global countries
    global age_buckets
    global sessions
    logging.warn('Loading data files')
    np.random.seed(9)
    train_full = pd.read_csv(TRAIN_DATA_FILE).sort_values('id')
    target_full = train_full[TARGET_COLUMN]
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( \
                                                      train_full[USER_COLUMNS] \
                                                      , train_full[TARGET_COLUMN] \
                                                      , test_size=TEST_SIZE \
                                                      , random_state=0)

    ## Read in data to predict for submission ##
    final_X_test = pd.read_csv(TEST_DATA_FINAL_FILE)

    ## Read supplemental datasets ##
    countries = pd.read_csv(COUNTRIES_FILE)
    age_buckets = pd.read_csv(AGE_GENDER_BUCKETS_FILE)

    ## Read session data ##
    sessions = pd.read_csv(SESSIONS_FILE)

# ### User data
def user_features(update_columns=True, newages=False):
    ''' Run feature engineering and cleaup for user features
    '''
    global train_full
    global final_X_test
    global CAT_COLS
    global NUM_COLS

    logging.warn('Processing user data features')
    train_full.index = train_full['id']
    final_X_test.index = final_X_test['id']
    target_full.index = train_full.index

    ## Change signup method for a random observation ##
    ## New signup method 'weibo' exists in final dataset ##
    train_full.loc[train_full.iloc[100,:]['id'],'signup_method'] = 'weibo'

    ## Clean up unreasonable values
    train_full.loc[train_full['age']>115,['age']] = np.nan
    final_X_test.loc[final_X_test['age']>115,['age']] = np.nan

    ## add new date features ##
    train_full['date_account_created'] = pd.to_datetime(train_full['date_account_created'])
    train_full['created_year'] = train_full['date_account_created'].dt.year
    train_full['created_month'] = train_full['date_account_created'].dt.month
    train_full['created_season'] = (train_full['created_month']-1) // 3
    train_full['created_day_of_week'] = train_full['date_account_created'].dt.dayofweek
    train_full['created_day_of_month'] = train_full['date_account_created'].dt.day
    train_full['created_part_of_month'] = (train_full['created_day_of_month']-1) // 4
    train_full['created_day_of_year'] = train_full['date_account_created'].dt.dayofyear
    train_full['created_days_ago'] = (DT.datetime.now() - train_full['date_account_created']).dt.days
    train_full['created_months_ago'] = (DT.datetime.now() \
                                            - train_full['date_account_created']).dt.days//30
    train_full['first_active_datetime'] = \
                        pd.to_datetime(train_full['timestamp_first_active'],format='%Y%m%d%H%M%S')
    train_full['first_active_hour'] = train_full['first_active_datetime'].dt.hour
    train_full['first_active_part_of_day'] = train_full['first_active_hour'] // 4

    ## repeat with final test ##
    final_X_test['date_account_created'] = pd.to_datetime(final_X_test['date_account_created'])
    final_X_test['created_year'] = final_X_test['date_account_created'].dt.year
    final_X_test['created_month'] = final_X_test['date_account_created'].dt.month
    final_X_test['created_season'] = (final_X_test['created_month']-1) // 3
    final_X_test['created_hour_of_day'] = final_X_test['date_account_created'].dt.hour
    final_X_test['created_day_of_week'] = final_X_test['date_account_created'].dt.dayofweek
    final_X_test['created_day_of_month'] = final_X_test['date_account_created'].dt.day
    final_X_test['created_part_of_month'] = (final_X_test['created_day_of_month']-1) // 4
    final_X_test['created_day_of_year'] = final_X_test['date_account_created'].dt.dayofyear
    final_X_test['created_days_ago'] = (DT.datetime.now() - final_X_test['date_account_created']).dt.days
    final_X_test['created_months_ago'] = (DT.datetime.now() \
                                            - final_X_test['date_account_created']).dt.days//30
    final_X_test['first_active_datetime'] = \
                        pd.to_datetime(final_X_test['timestamp_first_active'],format='%Y%m%d%H%M%S')
    final_X_test['first_active_hour'] = final_X_test['first_active_datetime'].dt.hour
    final_X_test['first_active_part_of_day'] = final_X_test['first_active_hour'] // 4

    ## Add new columns to model cols ##
    if update_columns:
        CAT_COLS += [
            'created_year',
            'created_month',
            'created_season',
            'created_day_of_week',
            'created_day_of_month',
            'created_part_of_month',
            'created_day_of_year',
            'first_active_hour',
            'first_active_part_of_day',
        ]
        NUM_COLS += [
            'created_days_ago',
            'created_months_ago',
        ]

    ## fill out missing ages with prediction model ##
    if newages:
        # Prepare for model creation
        mcl = MultiColumnLabelEncoder()
        mm = MinMaxScaler()
        ohe = OneHotEncoder()
        ss = StandardScaler(with_mean=False)
        ii = Imputer(strategy='most_frequent')
        ii2 = Imputer(strategy='mean')
        p1 = Pipeline([('mcl',mcl),('ii',ii),('ohe',ohe)])
        p2 = Pipeline([('ii',ii2),('ss',ss),('mm',mm)])

        # Create model
        nullages = np.isnan(train_full['age'])
        _ = p1.fit_transform(train_full[CAT_COLS])
        _ = p2.fit_transform(train_full[[n for n in NUM_COLS if n!='age']])
        X_1 = p1.transform(train_full[CAT_COLS][-nullages])
        X_2 = p2.transform(train_full[[n for n in NUM_COLS if n!='age']][-nullages])
        X = np.concatenate((X_1.todense(),X_2),axis=1)
        Y = train_full['age'][-nullages]
        en = ElasticNet(l1_ratio=1.0, alpha=0.001, normalize=True)
        en.fit(X,Y)

        # Add predicted ages column
        X = np.concatenate((
                p1.transform(train_full[CAT_COLS]).todense(),
                p2.transform(train_full[[n for n in NUM_COLS if n!='age']]))
            ,axis=1)
        train_full['age_pred'] = en.predict(X)

        # Prepare to make predictions on test data
        X_1 = p1.transform(final_X_test[CAT_COLS])
        X_2 = p2.transform(final_X_test[[n for n in NUM_COLS if n!='age']])
        X = np.concatenate((X_1.todense(),X_2),axis=1)
        final_X_test['age_pred'] = en.predict(X)

def calc_pca(cat_cols,num_cols,pca_n=5,prefix='pca_'):
    global NUM_COLS
    global train_full
    global final_X_test
    c = pca_n
    pca = PCA(n_components=c)
    ss = StandardScaler()
    mcl = MultiColumnLabelEncoder()
    ohe = OneHotEncoder()
    ss = StandardScaler(with_mean=False)
    ii = Imputer(strategy='most_frequent')
    ii2 = Imputer(strategy='mean')
    p1 = Pipeline([('mcl',mcl),('ii',ii),('ohe',ohe)])
    p2 = Pipeline([('ii',ii2),('ss',ss)])
    X_1 = p1.fit_transform(train_full[cat_cols])
    X_2 = p2.fit_transform(train_full[num_cols])
    X = np.concatenate((X_1.todense(),X_2),axis=1)
    tr_pca = pd.DataFrame( pca.fit_transform(X)
                        , columns = ['pca_collapsed_' + str(i) for i in range(c)]
                        , index = train_full.index
                    )
    X_1 = p1.transform(final_X_test[cat_cols])
    X_2 = p2.transform(final_X_test[num_cols])
    X = np.concatenate((X_1.todense(),X_2),axis=1)
    fnl_pca = pd.DataFrame( pca.transform(X)
                        , columns = ['pca_collapsed_' + str(i) for i in range(c)]
                        , index = final_X_test.index
                    )
    train_full = pd.concat([train_full,tr_pca],axis=1)
    final_X_test = pd.concat([final_X_test,fnl_pca],axis=1)
    logging.warn('PCA Explained variance: {}'.format(np.sum(pca.explained_variance_ratio_)))
    NUM_COLS += ['pca_collapsed_' + str(i) for i in range(c)]

def calc_lda(cat_cols,num_cols,prefix='lda_'):
    global NUM_COLS
    global train_full
    global final_X_test
    lda = LDA()
    ss = StandardScaler()
    mcl = MultiColumnLabelEncoder()
    ohe = OneHotEncoder()
    ss = StandardScaler(with_mean=False)
    ii = Imputer(strategy='most_frequent')
    ii2 = Imputer(strategy='mean')
    p1 = Pipeline([('mcl',mcl),('ii',ii),('ohe',ohe)])
    p2 = Pipeline([('ii',ii2),('ss',ss)])
    X_1 = p1.fit_transform(train_full[cat_cols])
    X_2 = p2.fit_transform(train_full[num_cols])
    X = np.concatenate((X_1.todense(),X_2),axis=1)
    Y = target_full['country_destination']
    c = len(set(Y))-1
    tr_lda = pd.DataFrame( lda.fit_transform(X, Y)
                        , columns = ['lda_collapsed_' + str(i) for i in range(c)]
                        , index = train_full.index
                    )
    X_1 = p1.transform(final_X_test[cat_cols])
    X_2 = p2.transform(final_X_test[num_cols])
    X = np.concatenate((X_1.todense(),X_2),axis=1)
    fnl_lda = pd.DataFrame( lda.transform(X)
                        , columns = ['lda_collapsed_' + str(i) for i in range(c)]
                        , index = final_X_test.index
                    )
    train_full = pd.concat([train_full,tr_lda],axis=1)
    final_X_test = pd.concat([final_X_test,fnl_lda],axis=1)
    NUM_COLS += ['lda_collapsed_' + str(i) for i in range(c)]

# ### Isolate components that are usable ##
def component_isolation(method='gradient',update_columns=False,add_pca=False,add_lda=False):
    ''' Determine usable features within categorical and/or numeric features
    '''
    global newcatcols
    global newnumcols
    global CAT_COLS
    global NUM_COLS

    if method=='rfe':
        abc = DecisionTreeClassifier()
        mcl = MultiColumnLabelEncoder() ; ohe = OneHotEncoder() ; im = Imputer(strategy='most_frequent')
        le = LabelEncoder()
        ndcg = make_scorer(ndcg_score, needs_proba=True, k=5)

        logging.warn('Recursively eliminating category features')
        ## Run first with category columns ##
        p = Pipeline([('im',im),('ohe',ohe)])
        X = p.fit_transform(
                mcl.fit_transform(train_full.iloc[:,:].loc[:,CAT_COLS])
            ) # trim original dataset to smaller sample
        Y = le.fit_transform(np.array(target_full.iloc[:,:]).ravel())
        rfe = RFECV(abc, scoring=ndcg, verbose=2, cv=2)
        rfe.fit( X , Y )
        logging.warn('Optimal number of user features: {}'.format(rfe.n_features_))
        ohe_features = p.named_steps['ohe'].active_features_[rfe.support_]
        ohe_indices = p.named_steps['ohe'].feature_indices_
        features = [i-1 for i in set(np.digitize(ohe_features,ohe_indices)) ]
        feature_names = list(train_full.loc[:,CAT_COLS].columns[features])
        logging.warn('Usable categorical features: \n\t{}'.format(str(feature_names)))
        if update_columns: CAT_COLS = feature_names
        newcatcols = feature_names

        logging.warn('Recursively eliminating numeric features')
        ## Run again with numeric columns ##
        im2 = Imputer(strategy='mean') ; le = LabelEncoder()
        X = im2.fit_transform(train_full.iloc[:,:].loc[:,NUM_COLS])
        Y = le.fit_transform(np.array(target_full.iloc[:,:]).ravel())
        rfe = RFECV(abc, scoring=ndcg, verbose=2, cv=2)
        rfe.fit( X , Y )
        logging.warn('Optimal number of user features: {}'.format(rfe.n_features_))
        features = rfe.support_
        feature_names = list(train_full.loc[:,NUM_COLS].columns[features])
        logging.warn('Usable numeric features: \n\t{}'.format(str(feature_names)))
        if update_columns: NUM_COLS = feature_names
        newnumcols = feature_names

    if method=='gradient':
        logging.warn('Finding feature importance values with GradientBoostingClassifier')

        ## Setup ##
        def gather_important_features(scores):
            importance = scores
            importance = np.array(sorted([ (k, importance[k]) for k in importance ], key=lambda x: x[1], reverse=True))
            importance = pd.DataFrame(importance,columns=['feature','fscore'])
            importance['fscore'] = importance['fscore'].astype('int') / importance['fscore'].astype('int').sum()
            importance.index = importance['feature']
            del importance['feature']
            m = np.min(importance['fscore'])
            importance['threshold'] = importance['fscore'] > m
            keep = [ int(i[1:]) for i in importance.loc[importance['threshold'],].index ]
            return keep

        xgb_params = {'learning_rate':0.1,
                            'objective':'multi:softprob','seed':0,'num_class':12,
                            'bst:colsample_bytree':0.25,'bst:max_depth':8,'bst:subsample':0.25,
                            'eval_metric':'ndcg'}
        n_runs = 200
        mcl = MultiColumnLabelEncoder() ; ohe = OneHotEncoder() ; im = Imputer(strategy='most_frequent')
        im2 = Imputer(strategy='mean')
        le = LabelEncoder()
        p1 = Pipeline([('mcl',mcl),('im',im),('ohe',ohe)])
        X_1 = p1.fit_transform(train_full[CAT_COLS])
        X_2 = im2.fit_transform(train_full[NUM_COLS])
        Y = le.fit_transform(np.array(target_full).ravel())

        ## Numeric columns ##
        logging.warn('Start with category features')
        dtrain = xgb.DMatrix(X_1, label=Y)
        gbdt = xgb.train(xgb_params, dtrain, n_runs, verbose_eval=100)
        keep = gather_important_features(gbdt.get_fscore())
        ohe_indices = p1.named_steps['ohe'].feature_indices_
        ohe_features = sorted(p1.named_steps['ohe'].active_features_[keep])
        features = [i-1 for i in set(np.digitize(ohe_features,ohe_indices)) ]
        feature_names = list(train_full.loc[:,CAT_COLS].columns[features])
        logging.warn('Usable categorical features: \n\t{}'.format(str(feature_names)))
        if update_columns: CAT_COLS = feature_names
        newcatcols = feature_names

        ## Numeric columns ##
        logging.warn('Run for numeric features')
        dtrain = xgb.DMatrix(X_2, label=Y)
        gbdt = xgb.train(xgb_params, dtrain, n_runs, verbose_eval=100)
        features = gather_important_features(gbdt.get_fscore())
        feature_names = list(train_full.loc[:,NUM_COLS].columns[features])
        logging.warn('Usable numeric features: \n\t{}'.format(str(feature_names)))
        if update_columns: NUM_COLS = feature_names
        newnumcols = feature_names

    if add_pca: calc_pca(newcatcols,newnumcols,5,'pca_minimized_')
    if add_lda: calc_lda(newcatcols,newnumcols,'lda_minimized_')

# #### age buckets
def attach_age_buckets(update_columns=True):
    ''' Merge user buckets data file
    '''
    global train_full
    global final_X_test
    global age_buckets

    logging.warn('Joining age bucket data')
    genders = set(age_buckets['gender'])
    age_buckets['age_merge'] = np.array([int(re.split(r'[-+]',str(x))[0]) \
                for x in age_buckets['age_bucket']]).astype('int')
    age_buckets.index = age_buckets['age_merge'].astype('string')\
            +'-'+age_buckets['country_destination']\
            +'-'+age_buckets['gender'].str.lower()

    am = sorted(set(age_buckets['age_merge']))
    tf = np.digitize(train_full['age'],am)
    fx = np.digitize(final_X_test['age'],am)
    dx = np.vectorize(lambda x: am[int(x)-1])
    tfdx = dx(tf)
    fxdx = dx(fx)
    for c in set(countries['country_destination']):
        for g in genders:
            z = train_full['gender'].str.lower()==g
            p = pd.Series(tfdx) \
                .astype('int') \
                .astype('string') \
                    +'-'+c \
                    +'-'+g
            p.index = train_full.index
            train_full.loc[z,'age_merge'+'-'+c+'-'+g] = p
            z = final_X_test['gender'].str.lower()==g
            p = pd.Series(fxdx) \
                .astype('int') \
                .astype('string') \
                    +'-'+c \
                    +'-'+g
            p.index = final_X_test.index
            final_X_test.loc[z,'age_merge'+'-'+c+'-'+g] = p

    age_buckets = age_buckets[[
            'age_merge' \
            ,'country_destination' \
            ,'gender' \
            ,'population_in_thousands']] \
        .groupby(['age_merge','country_destination','gender']).sum()

    age_buckets.index = pd.Series([ str(i[0])+'-'+i[1]+'-'+i[2] for i in age_buckets.index])

    for c in set(countries['country_destination']):
        for g in genders:
            z = train_full['gender'].str.lower()==g
            train_full = pd.merge(
                train_full \
                 , age_buckets \
                 , left_on=['age_merge'+'-'+c+'-'+g] \
                 , right_index=True \
                 , left_index=False \
                 , how='left'
            )
            train_full.rename(columns={'population_in_thousands':\
                                        'population_in_thousands'+c+g}, inplace=True)
            z = final_X_test['gender'].str.lower()==g
            final_X_test = pd.merge(
                final_X_test \
                 , age_buckets \
                 , left_on=['age_merge'+'-'+c+'-'+g] \
                 , right_index=True \
                 , left_index=False \
                 , how='left'
            )
            final_X_test.rename(columns={'population_in_thousands':\
                                          'population_in_thousands'+c+g}, inplace=True)

    if update_columns:
        global NUM_COLS
        NUM_COLS += [ p+g for p,g in product([ 'population_in_thousands'+c \
                           for c in set(countries['country_destination']) ],genders) ]

# #### Sessions
def attach_sessions(collapse=True,pca=True, lm=True, update_columns=True, pca_n=5, session_columns=[]):
    ''' Collapse and merge user session data
    '''
    global train_full
    global final_X_test
    global sessions
    global NUM_COLS

    logging.warn('Processing session data model')
    cf = ['action','action_type','action_detail','device_type']
    s = sessions[cf].copy().fillna('missing')
    mcl = MultiColumnLabelEncoder()
    ohe = OneHotEncoder()
    x = ohe.fit_transform(
        mcl.fit_transform(s)
    )
    n = x.shape[1]//50 # 50 chunks of session data

    ## Function to minimize session data at the user level ##
    def minimize_df(i,lim,n,nparr):
        m = min(lim,(i+1)*n)
        return pd.DataFrame(nparr[:,i*n:m].toarray() \
                    ,index=sessions['user_id']) \
                    .groupby(level=0).sum()

    ## Minimize session data ##
    z = ( minimize_df(y,x.shape[1],50,x) for y in range(n+1) )

    ## Combine datasets ##
    sessions_new = pd.concat(z,axis=1)
    sessions_new.columns = [ 'session_'+str(i) for i in range(len(sessions_new.columns)) ]

    ## Add session features to training data ##
    train_full = pd.merge(train_full, sessions_new, how='left', left_index=True, right_index=True)
    final_X_test = pd.merge(final_X_test, sessions_new, how='left', left_index=True, right_index=True)

    ## Set sessions to zero for users with no usage ##
    train_full.loc[:,sessions_new.columns] = train_full.loc[:,sessions_new.columns].fillna(0)
    final_X_test.loc[:,sessions_new.columns] = final_X_test.loc[:,sessions_new.columns].fillna(0)

    ## Add per day calculations
    train_full['session_total'] = 0
    final_X_test['session_total'] = 0

    subsess = re.compile(r'session_')

    for s in sessions_new.columns:
        ## Per day
        new = subsess.sub('sessionperday_',s)
        train_full[new] = train_full[s] / train_full['created_days_ago']
        final_X_test[new] = final_X_test[s] / final_X_test['created_days_ago']
        if update_columns: NUM_COLS += [new]

        ## Total
        train_full['session_total'] += train_full[s]
        final_X_test['session_total'] += final_X_test[s]

    train_full['sessionperday_total'] = train_full['session_total'] / train_full['created_days_ago']
    final_X_test['sessionperday_total'] = final_X_test['session_total'] / final_X_test['created_days_ago']
    if update_columns: NUM_COLS += ['session_total','sessionperday_total']

    ## Prepare data for feature extraction ##
    target = pd.DataFrame({'country_destination':train_full['country_destination']})
    target.index = train_full['id']
    merged = pd.merge(\
                sessions_new\
                , target\
                , how='inner'\
                , left_index=True
                , right_index=True
             )

    if collapse:
        ## Extract most importance features ##
        logging.warn('Extracting meaningful session features')
        abc = DecisionTreeClassifier()
        ndcg = make_scorer(ndcg_score, needs_proba=True, k=5)
        rfe = RFECV(abc, scoring=ndcg, verbose=2, cv=2)
        le = LabelEncoder()
        X = np.array(merged)[:,:-1]
        Y = le.fit_transform(np.array(merged)[:,-1:].ravel())
        rfe.fit( X , Y )
        features = rfe.support_
        fi = rfe.ranking_
        logging.warn('Optimal number of session features: {}'.format(rfe.n_features_))
        session_columns = list(sessions_new.iloc[:,features].columns)
        logging.warn('Meaningful columns: \n\r{}'.format('\n\t'.join(session_columns)))
    else:
        if not session_columns:
            session_columns = list(sessions_new.columns)
    if update_columns: NUM_COLS += session_columns

    ## PCA ##
    if pca:
        logging.warn('Collapse session features with PCA')
        c = pca_n if pca_n else len(session_columns)
        pca = PCA(n_components=c)
        ss = StandardScaler()
        tr_pca = pd.DataFrame( pca.fit_transform(ss.fit_transform(train_full.loc[:,session_columns])) \
                            , columns = ['pca_session_' + str(i) for i in range(c)] \
                            , index = train_full.index \
                        )
        fnl_pca = pd.DataFrame( pca.transform(ss.transform(final_X_test.loc[:,session_columns])) \
                            , columns = ['pca_session_' + str(i) for i in range(c)]
                            , index = final_X_test.index \
                        )
        logging.warn('PCA Explained variance: {}'.format(np.sum(pca.explained_variance_ratio_)))
        train_full = pd.concat([train_full,tr_pca],axis=1)
        final_X_test = pd.concat([final_X_test,fnl_pca],axis=1)
        if update_columns: NUM_COLS += ['pca_session_' + str(i) for i in range(c)]

    if lm:
        ## Create prediction model for PCA features ##
        logging.warn('Creating regression model for session features')
        components = len(session_columns)

        ## Split out category and numeric columns ##
        tr_cat = train_full.loc[:,CAT_COLS]
        tr_cat.index = train_full['id']
        tr_num = train_full.loc[:,NUM_COLS]
        tr_num.index = train_full['id']

        final_tst_cat = final_X_test.loc[:,CAT_COLS]
        final_tst_cat.index = final_X_test['id']
        final_tst_num = final_X_test.loc[:,NUM_COLS]
        final_tst_num.index = final_X_test['id']

        ## Merge with new features ##
        nonzeros = train_full['session_total'] > 0
        merged_cats = pd.merge(tr_cat \
                                , train_full.loc[nonzeros,session_columns] \
                                , how='inner' \
                                , left_index=True \
                                , right_index=True  )
        merged_nums = pd.merge(tr_num \
                                , train_full.loc[nonzeros,session_columns] \
                                , how='inner' \
                                , left_index=True \
                                , right_index=True  )
        mcl = MultiColumnLabelEncoder()
        mm = MinMaxScaler()
        ohe = OneHotEncoder()
        ss = StandardScaler(with_mean=False)
        ii = Imputer(strategy='most_frequent')
        ii2 = Imputer(strategy='mean')
        p1 = Pipeline([('mcl',mcl),('ii',ii),('ohe',ohe)])
        p2 = Pipeline([('ii',ii2),('ss',ss),('mm',mm)])

        trcat_transformed = p1.fit_transform(tr_cat).todense()
        trnum_transformed = p2.fit_transform(tr_num)
        trcombined = np.concatenate((trcat_transformed, trnum_transformed), axis=1)

        tstcat_final_transformed = p1.transform(final_tst_cat).todense()
        tstnum_final_transformed = p2.transform(final_tst_num)
        tstcombined_final = np.concatenate((tstcat_final_transformed, tstnum_final_transformed), axis=1)

        mcat_transformed = p1.transform(merged_cats.iloc[:,:-1*components]).todense()
        mnum_transformed = p2.transform(merged_nums.iloc[:,:-1*components])
        mcombined = np.concatenate((mcat_transformed, mnum_transformed), axis=1)
        #
        # lm_cvs = [ ElasticNetCV( \
        #             l1_ratio=[.1, .5, .7, .9, .95, 1] \
        #             , alphas=[0.001,0.01,0.05,0.1,0.5,0.9] \
        #             , max_iter=1000, n_jobs=2
        #         ) \
        #         for l in np.arange(components) ]
        # for i,lm in enumerate(lm_cvs):
        #     lm.fit(mcombined, merged_cats.iloc[:,merged_cats.shape[1]-i-1])

        # for l in lm_cvs: logging.warn('L1: {} Alpha: {}'.format(l.l1_ratio_,l.alpha_))
        # lms = [ ElasticNet(l1_ratio=l.l1_ratio_, alpha=l.alpha_, normalize=True) \
                    # for l in lm_cvs ]
        lms = [ ElasticNet(l1_ratio=1.0, alpha=0.001, normalize=True) \
                    for l in np.arange(components) ]

        for i,lm in enumerate(lms):
            lm.fit(mcombined, merged_cats.iloc[:,merged_cats.shape[1]-i-1])
            train_full.loc[:,'lm_'+str(i)] = lm.predict(trcombined)
            final_X_test.loc[:,'lm_'+str(i)] = lm.predict(tstcombined_final)
            lms[i] = lm

        # merged_tst = pd.merge(X_test \
        #                         , lm_features \
        #                         , how='inner' \
        #                         , left_index=True \
        #                         , right_index=True  )
        # for i in range(components):
        #     logging.warn('MSE {}: {}'.format(i \
        #         , np.sqrt(np.mean(np.sum((merged_tst['lm_'+str(i)] \
        #                                         - merged_tst[i])**2))) \
        #     ))
            # train_full.loc[merged_nums.index,'lm_'+str(i)] = merged_nums[i]
            # X_test.loc[merged_nums_tst.index,'lm_'+str(i)] = merged_nums_tst[i]

        if update_columns: NUM_COLS += [ 'lm_'+str(i) for i in range(components) ]

        ## Add per day calculations
        train_full['lm_total'] = 0
        final_X_test['lm_total'] = 0

        sublm = re.compile(r'lm_')

        for s in [ 'lm_'+str(i) for i in range(components) ]:
            ## Per day
            new = sublm.sub('lmperday_',s)
            train_full[new] = train_full[s] / train_full['created_days_ago']
            final_X_test[new] = final_X_test[s] / final_X_test['created_days_ago']
            if update_columns: NUM_COLS += [new]

            ## Total
            train_full['lm_total'] += train_full[s]
            final_X_test['lm_total'] += final_X_test[s]

        ## Per day total
        train_full['lmperday_total'] = train_full['lm_total'] / train_full['created_days_ago']
        final_X_test['lmperday_total'] = final_X_test['lm_total'] / final_X_test['created_days_ago']
        if update_columns: NUM_COLS += ['lm_total','lmperday_total']

def compile_nn():
    nn_model = Sequential()
    nn_model.add(Dense(output_dim=24, input_dim=X.shape[1], \
        init="glorot_uniform", W_regularizer=l2(0.01)))
    nn_model.add(Activation("softmax"))
    nn_model.add(Dense(output_dim=Y.shape[1], input_dim=24))
    nn_model.add(Activation("softmax"))
    nn_model.add(Activation("relu"))
    nn_model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return nn_model

# ### Run final model ##
def final_model(test=True,grid_cv=False,save_final_results=True):
    ''' execute final model
    '''
    global train_full
    global target_full
    global X_train
    global X_test
    global Y_train
    global Y_test
    global final_X_test
    global GS_CV

    logging.warn('Create boosted trees model with training data')
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
    Y = cat_full

    if test:
        ## Set up X,Y data for modeling ##
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( \
                                                          train_full[CAT_COLS+NUM_COLS] \
                                                          , target_full \
                                                          , test_size=TEST_SIZE \
                                                          , random_state=0)
        cat_le = le.transform(np.array(Y_train).ravel())
        cat_tst_le = le.transform(np.array(Y_test).ravel())
        cat_lb = lb.transform(np.array(Y_train).ravel())
        cat_tst_lb = lb.transform(np.array(Y_test).ravel())
        X_train = np.concatenate((p.transform(X_train[CAT_COLS]).todense() \
                                ,im2.transform(np.array(X_train[NUM_COLS]))),axis=1)
        X_test = np.concatenate((p.transform(X_test[CAT_COLS]).todense() \
                                ,im2.transform(np.array(X_test[NUM_COLS]))),axis=1)

        ## Run model with only training data ##
        logging.warn('Running model with training data')
        xgb = XGBClassifier(learning_rate=0.01, n_estimators=50,
                            objective='multi:softprob',seed=0, **GS_CV)
        xgb.fit(X_train , cat_le)

        ## Run model with only training data ##
        logging.warn('Test prediction accuracy')
        p_pred = xgb.predict(X_test)
        p_pred_i = le.inverse_transform(p_pred)
        p_pred_p = xgb.predict_proba(X_test)
        logging.warn('Accuracy: '+str(np.mean(p_pred_i == np.array(Y_test).ravel())))
        logging.warn('\n'+classification_report(p_pred_i,np.array(Y_test).ravel()))
        logging.warn('Log Loss: {}'.format(log_loss(np.array(Y_test).ravel(), p_pred_p)))
        logging.warn('Label Ranking Precision score: {}'\
                        .format(label_ranking_average_precision_score(cat_tst_lb, p_pred_p)))
        logging.warn('Label Ranking loss: {}'.format(label_ranking_loss(cat_tst_lb, p_pred_p)))
        logging.warn('NDCG score: {}'.format(ndcg_score(cat_tst_lb, p_pred_p, k=5)))

    if grid_cv:
        ## Run grid search to find optimal parameters ##
        params_grid = {
        		         'max_depth':[ 15, 20, 25 ] ,
                         'subsample':[ 0.25, 0.5  ] ,
                        #  'colsample_bytree':[ 0.25, 0.5, 0.75 ] ,
                }
        logging.warn('Running grid search CV with params: {}'.format(params_grid))
        ndcg = make_scorer(ndcg_score, needs_proba=True, k=5)
        xgb = XGBClassifier(n_estimators=50, objective='multi:softprob', seed=0)
        cv = GridSearchCV(xgb, params_grid, scoring=ndcg).fit(X, Y)
        logging.warn('Best XGB params: {}'.format(cv.best_params_))
        GS_CV = cv.best_params_

    ## Run model with all data and save ##
    if save_final_results:
        ''' Write results to a csv file
            NOTE: sorting is not done here
        '''
        logging.warn('Make predictions for final test set')
        logging.warn('Running model with all training data')
        xgb = XGBClassifier(learning_rate=0.01, n_estimators=500,
                            objective='multi:softprob',seed=0, **GS_CV)
        xgb.fit(X , Y)
        X = np.concatenate((p.transform(final_X_test[CAT_COLS]).todense() \
                                ,im2.transform(np.array(final_X_test[NUM_COLS]))),axis=1)
        f_pred = xgb.predict_proba(X)

        ## Write to submission file ##
        k = 5
        results = np.sort(f_pred)[:,::-1][:,:k].ravel()
        labels = le.inverse_transform(np.argsort(f_pred)[:,::-1][:,:k].ravel())
        ids = np.array(final_X_test['id'])
        ids = np.array([ ids for i in range(k) ]).T.ravel()
        results_df = pd.DataFrame({'id':ids})
        results_df['country'] = labels
        results_df.to_csv('Data/submission.csv',index=False)

def neural_model(test=True,save_final_results=True):
    global train_full
    global target_full
    global X_train
    global X_test
    global Y_train
    global Y_test
    global final_X_test
    global GS_CV

    logging.warn('Create neural model with training data')
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
    Y = cat_full_lb

    ## Neural Network ##
    model = compile_nn()

    if test:
        ## Set up X,Y data for modeling ##
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( \
                                                          train_full[CAT_COLS+NUM_COLS] \
                                                          , target_full \
                                                          , test_size=TEST_SIZE \
                                                          , random_state=0)
        cat_lb = lb.transform(np.array(Y_train).ravel())
        cat_tst_lb = lb.transform(np.array(Y_test).ravel())
        X_train = np.concatenate((p.transform(X_train[CAT_COLS]).todense() \
                                ,im2.transform(np.array(X_train[NUM_COLS]))),axis=1)
        X_test = np.concatenate((p.transform(X_test[CAT_COLS]).todense() \
                                ,im2.transform(np.array(X_test[NUM_COLS]))),axis=1)

        ## Run model with only training data ##
        logging.warn('Running model with training data')
        model.fit(X_train, cat_lb, batch_size=128, nb_epoch=10,
            validation_data=(X_test, cat_tst_lb))

        ## Run model with only training data ##
        logging.warn('Test prediction accuracy')
        p_pred = model.predict(X_test)
        p_pred_i = lb.inverse_transform(p_pred)
        p_pred_p = model.predict_proba(X_test)
        logging.warn('Accuracy: '+str(np.mean(p_pred_i == np.array(Y_test).ravel())))
        logging.warn('\n'+classification_report(p_pred_i,np.array(Y_test).ravel()))
        logging.warn('Log Loss: {}'.format(log_loss(np.array(Y_test).ravel(), p_pred_p)))
        logging.warn('Label Ranking Precision score: {}'\
                        .format(label_ranking_average_precision_score(cat_tst_lb, p_pred_p)))
        logging.warn('Label Ranking loss: {}'.format(label_ranking_loss(cat_tst_lb, p_pred_p)))
        logging.warn('NDCG score: {}'.format(ndcg_score(cat_tst_lb, p_pred_p, k=5)))

    if save_final_results:
        ''' Write results to a csv file
            NOTE: sorting is not done here
        '''
        logging.warn('Make predictions for final test set')
        logging.warn('Running model with all training data')
        model.fit(X, Y, nb_epoch=25, batch_size=128)
        X = np.concatenate((p.transform(final_X_test[CAT_COLS]).todense() \
                                ,im2.transform(np.array(final_X_test[NUM_COLS]))),axis=1)
        f_pred = model.predict_proba(X)

        ## Write to submission file ##
        k = 5
        results = np.sort(f_pred)[:,::-1][:,:k].ravel()
        labels = le.inverse_transform(np.argsort(f_pred)[:,::-1][:,:k].ravel())
        ids = np.array(final_X_test['id'])
        ids = np.array([ ids for i in range(k) ]).T.ravel()
        results_df = pd.DataFrame({'id':ids})
        results_df['country'] = labels
        results_df.to_csv('Data/submission.csv',index=False)

def combi_model(test=True,save_final_results=True):
    global train_full
    global target_full
    global X_train
    global X_test
    global Y_train
    global Y_test
    global final_X_test
    global GS_CV

    logging.warn('Create neural model with training data')

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
    X_p = np.concatenate((p.transform(final_X_test[CAT_COLS]).todense() \
                            ,im2.transform(np.array(final_X_test[NUM_COLS]))),axis=1)
    Y = cat_full_lb

    ## Neural Network ##
    nn_model = compile_nn()

    ## NN weighting mult ##
    mix = 0.1

    if save_final_results:
        ''' Write results to a csv file
            NOTE: sorting is not done here
        '''
        logging.warn('Make predictions for final test set')
        logging.warn('Running model with all training data')

        nn_model.fit(X, Y, nb_epoch=25, batch_size=128)
        xgb = XGBClassifier(learning_rate=0.01, n_estimators=500,
                            objective='multi:softprob',seed=0, **GS_CV)
        Y = cat_full
        xgb.fit(X , Y)

        ## Make predictions ##
        f_pred_nn = model.predict_proba(X_p)
        f_pred_xgb = xgb.predict_proba(X_p)

        ## merge models ##
        f_pred_nn *= mix
        f_pred_xgb *= (1-mix)
        f_pred = f_pred_nn + f_pred_xgb

        ## Write to submission file ##
        k = 5
        results = np.sort(f_pred)[:,::-1][:,:k].ravel()
        labels = le.inverse_transform(np.argsort(f_pred)[:,::-1][:,:k].ravel())
        ids = np.array(final_X_test['id'])
        ids = np.array([ ids for i in range(k) ]).T.ravel()
        results_df = pd.DataFrame({'id':ids})
        results_df['country'] = labels
        results_df.to_csv('Data/submission.csv',index=False)

def run():
    global NUM_COLS
    global CAT_COLS
    declare_args(); load_data()
    user_features(update_columns=True, newages=True)
    attach_age_buckets(update_columns=True)
    attach_sessions(collapse=False, pca=True, lm=True, update_columns=True, pca_n=250)
    # component_isolation(method='gradient', update_columns=False, add_pca=False, add_lda=False)
    final_model(test=False, grid_cv=False, save_final_results=True)
    # neural_model(test=False, save_final_results=True)
    # combi_model(test=False,save_final_results=True)

# if __name__=='__main__':
#     run()
