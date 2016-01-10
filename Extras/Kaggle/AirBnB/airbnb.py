
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn import cross_validation

# Metrics
from sklearn.metrics import log_loss, classification_report \
                            , label_ranking_average_precision_score, label_ranking_loss

# Neural Nets
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU

# XGBoost
from xgboost.sklearn import XGBClassifier

# matplotlib
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

# Grid Search
from sklearn.grid_search import GridSearchCV

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
def user_features(update_columns=True):
    ''' Run feature engineering and cleaup for user features
    '''
    global train_full
    global final_X_test

    logging.warn('Processing user data features')
    train_full.index = train_full['id']
    final_X_test.index = final_X_test['id']

    ## Change signup method for a random observation ##
    ## New signup method 'weibo' exists in final dataset ##
    train_full.loc[train_full.iloc[100,:]['id'],'signup_method'] = 'weibo'

    ## Clean up unreasonable values
    train_full.loc[train_full['age']>115,['age']] = np.nan
    final_X_test.loc[final_X_test['age']>115,['age']] = np.nan

    ## add new date features ##
    train_full['date_account_created'] = pd.to_datetime(train_full['date_account_created'])
    train_full['date_first_booking'] = pd.to_datetime(train_full['date_first_booking'])
    train_full['year_created'] = train_full['date_account_created'].dt.year
    train_full['month_created'] = train_full['date_account_created'].dt.month
    train_full['year_first_booking'] = train_full['date_first_booking'].dt.year
    train_full['month_first_booking'] = train_full['date_first_booking'].dt.month
    train_full['days_to_first_booking'] = train_full['date_first_booking']-train_full['date_account_created']
    train_full['days_ago_created'] = (DT.datetime.now() - train_full['date_account_created']).dt.days
    train_full['days_ago_first_booking'] = (DT.datetime.now() - train_full['date_first_booking']).dt.days

    ## repeat with final test ##
    final_X_test['date_account_created'] = pd.to_datetime(final_X_test['date_account_created'])
    final_X_test['date_first_booking'] = pd.to_datetime(final_X_test['date_first_booking'])
    final_X_test['year_created'] = final_X_test['date_account_created'].dt.year
    final_X_test['month_created'] = final_X_test['date_account_created'].dt.month
    final_X_test['year_first_booking'] = final_X_test['date_first_booking'].dt.year
    final_X_test['month_first_booking'] = final_X_test['date_first_booking'].dt.month
    final_X_test['days_to_first_booking'] = final_X_test['date_first_booking']-final_X_test['date_account_created']
    final_X_test['days_ago_created'] = (DT.datetime.now() - final_X_test['date_account_created']).dt.days
    final_X_test['days_ago_first_booking'] = (DT.datetime.now() - final_X_test['date_first_booking']).dt.days

    ## Add new columns to model cols ##
    if update_columns:
        global CAT_COLS
        global NUM_COLS
        CAT_COLS += [
            'year_first_booking',
            'month_first_booking',
            'year_created',
            'month_created',
        ]
        NUM_COLS += [
            'days_to_first_booking',
            'days_ago_created',
            'days_ago_first_booking',
        ]

    ## Add special values for nulls for categorical fields
    train_full['year_first_booking'].fillna(1970,inplace=True)
    train_full['month_first_booking'].fillna(1970,inplace=True)
    train_full['date_first_booking'].fillna(pd.to_datetime('1970-01-01'),inplace=True)

    ## Repeat with final test ##
    final_X_test['year_first_booking'].fillna(1970,inplace=True)
    final_X_test['month_first_booking'].fillna(0,inplace=True)
    final_X_test['date_first_booking'].fillna(pd.to_datetime('1970-01-01'),inplace=True)

    ## add new date features ##
    train_full.loc[train_full['days_to_first_booking']<pd.Timedelta(0),['days_to_first_booking']] = np.nan
    train_full['days_to_first_booking'] = train_full['days_to_first_booking'].astype('timedelta64[D]')

    final_X_test.loc[final_X_test['days_to_first_booking']<pd.Timedelta(0),['days_to_first_booking']] = np.nan
    final_X_test['days_to_first_booking'] = final_X_test['days_to_first_booking'].astype('timedelta64[D]')


# ### Isolate components that are usable ##
def component_isolation(categorical=True,numeric=True,update_columns=False):
    ''' Determine usable features within categorical and/or numeric features
    '''
    abc = AdaBoostClassifier(learning_rate=0.1)
    mcl = MultiColumnLabelEncoder() ; ohe = OneHotEncoder() ; im = Imputer(strategy='most_frequent')
    le = LabelEncoder()

    if categorical:
        global CAT_COLS
        logging.warn('Recursively eliminating category features')
        ## Run first with category columns ##
        p = Pipeline([('im',im),('ohe',ohe)])
        X = p.fit_transform(
                mcl.fit_transform(train_full.iloc[:,:].loc[:,CAT_COLS])
            ) # trim original dataset to smaller sample
        Y = le.fit_transform(np.array(target_full.iloc[:,:]).ravel())
        rfe = RFECV(abc, scoring='log_loss', verbose=2, cv=2)
        rfe.fit( X , Y )
        logging.warn('Optimal number of user features: {}'.format(rfe.n_features_))
        ohe_features = p.named_steps['ohe'].active_features_[rfe.support_]
        ohe_indices = p.named_steps['ohe'].feature_indices_
        features = [i-1 for i in set(np.digitize(ohe_features,ohe_indices)) ]
        feature_names = list(train_full.loc[:,CAT_COLS].columns[features])
        logging.warn('Usable categorical features: \n\t{}'.format('\n\t'.join(feature_names)))
        if update_columns: CAT_COLS = feature_names

    if numeric:
        global NUM_COLS
        logging.warn('Recursively eliminating numeric features')
        ## Run again with numeric columns ##
        im2 = Imputer(strategy='mean') ; le = LabelEncoder()
        X = im2.fit_transform(train_full.iloc[:,:].loc[:,NUM_COLS])
        Y = le.fit_transform(np.array(target_full.iloc[:,:]).ravel())
        rfe = RFECV(abc, scoring='log_loss', verbose=2, cv=2)
        rfe.fit( X , Y )
        logging.warn('Optimal number of user features: {}'.format(rfe.n_features_))
        features = rfe.support_
        feature_names = list(train_full.loc[:,NUM_COLS].columns[features])
        logging.warn('Usable numeric features: \n\t{}'.format('\n\t'.join(feature_names)))
        if update_columns: NUM_COLS = feature_names

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
    for c in set(countries['country_destination']):
        for g in genders:
            train_full['age_merge'+'-'+c+'-'+g] = \
                                pd.Series(dx(tf)) \
                                    .astype('int') \
                                    .astype('string') \
                                +'-'+c \
                                +'-'+g
            final_X_test['age_merge'+'-'+c+'-'+g] = \
                                pd.Series(dx(fx)) \
                                    .astype('int') \
                                    .astype('string') \
                                +'-'+c \
                                +'-'+g

    age_buckets = age_buckets[[
            'age_merge' \
            ,'country_destination' \
            ,'gender' \
            ,'population_in_thousands']] \
        .groupby(['age_merge','country_destination','gender']).sum()

    age_buckets.index = pd.Series([ str(i[0])+'-'+i[1]+'-'+i[2] for i in age_buckets.index])

    for c in set(countries['country_destination']):
        for g in genders:
            train_full = pd.merge(
                train_full \
                 , age_buckets \
                 , left_on=['age_merge'+'-'+c+'-'+g] \
                 , right_index=True \
                 , how='left' \
                 , suffixes=(c+g,c+g)
            )
            final_X_test = pd.merge(
                final_X_test \
                 , age_buckets \
                 , left_on=['age_merge'+'-'+c+'-'+g] \
                 , right_index=True \
                 , how='left' \
                 , suffixes=(c+g,c+g)
            )

    if update_columns:
        global NUM_COLS
        NUM_COLS += [ p+g for p,g in product([ 'population_in_thousands'+c for c in set(countries['country_destination']) ],genders) ]

# #### Sessions
def attach_sessions(collapse=True,pca=True, lm=True, update_columns=True, pca_n=5):
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
    n = x.shape[1]//50

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
        abc = AdaBoostClassifier(learning_rate=0.1)
        rfe = RFECV(abc, scoring='log_loss', verbose=2, cv=2)
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
        # session_columns = list(sessions_new.columns)
        ## Already ran ## 
        session_columns = ['session_14',
                             'session_15',
                             'session_16',
                             'session_17',
                             'session_18',
                             'session_19',
                             'session_20',
                             'session_21',
                             'session_22',
                             'session_23',
                             'session_24',
                             'session_25',
                             'session_26',
                             'session_27',
                             'session_28',
                             'session_29',
                             'session_30',
                             'session_31',
                             'session_32',
                             'session_33',
                             'session_34',
                             'session_35',
                             'session_36',
                             'session_37',
                             'session_38',
                             'session_39',
                             'session_40',
                             'session_41',
                             'session_42',
                             'session_43',
                             'session_44',
                             'session_45',
                             'session_46',
                             'session_47',
                             'session_48',
                             'session_49',
                             'session_50',
                             'session_51',
                             'session_52',
                             'session_53',
                             'session_54',
                             'session_55',
                             'session_85',
                             'session_163',
                             'session_174',
                             'session_175',
                             'session_176',
                             'session_197',
                             'session_200',
                             'session_213',
                             'session_226',
                             'session_233',
                             'session_300',
                             'session_331',
                             'session_332',
                             'session_333',
                             'session_334',
                             'session_335',
                             'session_336',
                             'session_337',
                             'session_338',
                             'session_339',
                             'session_340',
                             'session_341',
                             'session_342',
                             'session_343',
                             'session_344',
                             'session_345',
                             'session_346',
                             'session_347',
                             'session_348',
                             'session_349',
                             'session_350',
                             'session_351',
                             'session_352',
                             'session_353',
                             'session_354',
                             'session_355',
                             'session_356',
                             'session_357',
                             'session_358',
                             'session_359',
                             'session_360',
                             'session_361',
                             'session_362',
                             'session_363',
                             'session_364',
                             'session_365',
                             'session_366',
                             'session_369',
                             'session_409',
                             'session_413',
                             'session_414',
                             'session_415',
                             'session_417',
                             'session_419',
                             'session_425',
                             'session_427',
                             'session_430',
                             'session_435',
                             'session_441',
                             'session_445',
                             'session_459',
                             'session_460',
                             'session_461',
                             'session_462',
                             'session_463']
    if update_columns: NUM_COLS += session_columns

    ## PCA ##
    if pca:
        logging.warn('Collapse session features with PCA')
        c = pca_n
        pca = PCA(n_components=c)
        ss = StandardScaler()
        tr_pca = pd.DataFrame( pca.fit_transform(ss.fit_transform(train_full.loc[:,session_columns])) \
                            , columns = ['pca_session_' + str(i) for i in range(c)] \
                            , index = train_full.index \
                        )
        fnl_pca = pd.DataFrame( ss.fit_transform(pca.transform(final_X_test.loc[:,session_columns])) \
                            , columns = ['pca_session_' + str(i) for i in range(c)]
                            , index = final_X_test.index \
                        )
        logging.warn('PCA Explained variance: {}'.format(np.sum(pca.explained_variance_ratio_)))
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
            merged_cats = pd.merge(tr_cat \
                                    , tr_pca \
                                    , how='inner' \
                                    , left_index=True \
                                    , right_index=True  )
            merged_nums = pd.merge(tr_num \
                                    , tr_pca \
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

            lm_cvs = [ ElasticNetCV( \
                        l1_ratio=[.1, .5, .7, .9, .95, 1] \
                        , alphas=[0.001,0.01,0.05,0.1,0.5,0.9] \
                        , max_iter=1000, n_jobs=2
                    ) \
                    for l in np.arange(components) ]
            for i,lm in enumerate(lm_cvs):
                lm.fit(mcombined, merged_cats.iloc[:,merged_cats.shape[1]-i-1])

            for l in lm_cvs: logging.warn('L1: {} Alpha: {}'.format(l.l1_ratio_,l.alpha_))
            lms = [ ElasticNet(l1_ratio=l.l1_ratio_, alpha=l.alpha_, normalize=True) \
                        for l in lm_cvs ]

            for i,lm in enumerate(lms):
                lm.fit(mcombined, merged_cats.iloc[:,merged_cats.shape[1]-i-1])
                train_full.loc[:,'lm_'+str(i)] = lm.predict(trcombined)
                final_X_test.loc[:,'lm_'+str(i)] = lm.predict(tstcombined_final)
                lms[i] = lm

            merged_tst = pd.merge(X_test \
                                    , lm_features \
                                    , how='inner' \
                                    , left_index=True \
                                    , right_index=True  )
            for i in range(components):
                logging.warn('MSE {}: {}'.format(i \
                    , np.sqrt(np.mean(np.sum((merged_tst['lm_'+str(i)] \
                                                    - merged_tst[i])**2))) \
                ))
                # train_full.loc[merged_nums.index,'lm_'+str(i)] = merged_nums[i]
                # X_test.loc[merged_nums_tst.index,'lm_'+str(i)] = merged_nums_tst[i]

            train_full = pd.concat([train_full,tr_pca],axis=1)
            X_test = pd.concat([X_test,tst_pca],axis=1)
            final_X_test = pd.concat([final_X_test,fnl_pca],axis=1)

            if update_columns: NUM_COLS += [ 'lm_'+str(i) for i in range(components) ]

# ### Run final model ##
def final_model(test=True,grid_cv=False,save_results=True):
    ''' execute final model
    '''
    global train_full
    global target_full
    global X_train
    global X_test
    global Y_train
    global Y_test
    global final_X_test

    logging.warn('Create boosted trees model with training data')
    ## Encode categories ##
    le = LabelEncoder()
    cat_full = le.fit_transform(np.array(target_full).ravel())

    mcl = MultiColumnLabelEncoder() ; ohe = OneHotEncoder() ; im = Imputer(strategy='most_frequent')
    im2 = Imputer(strategy='mean')
    p = Pipeline([('mcl',mcl),('im',im),('ohe',ohe)])

    ## full dataset ##
    X = np.concatenate((p.fit_transform(train_full[CAT_COLS]).todense() \
                            ,im2.fit_transform(np.array(train_full[NUM_COLS]))),axis=1)
    Y = cat_full

    if test:
        ## Set up X,Y data for modeling ##
        cat_le = le.transform(np.array(Y_train).ravel())
        cat_tst_le = le.transform(np.array(Y_test).ravel())
        X_train = np.concatenate((p.transform(X_train[CAT_COLS]).todense() \
                                ,im2.transform(np.array(X_train[NUM_COLS]))),axis=1)
        X_test = np.concatenate((p.transform(X_test[CAT_COLS]).todense() \
                                ,im2.transform(np.array(X_test[NUM_COLS]))),axis=1)
        Y_train = cat_le
        Y_test = cat_tst_le

        if grid_cv:
            ''' ## Run grid search to find optimal parameters ##
            params_grid = {'learning_rate':[0.3,0.1,0.05,0.02,0.01]
            		, 'max_depth':[ 4, 6, 8 ]}
            xgb = XGBClassifier(n_estimators=100, objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
            gs_csv = GridSearchCV(xgb, params_grid).fit(X_train, Y_train)
            print(gs_csv.best_params_)
            '''

        ## Run model with only training data ##
        logging.warn('Running model with training data')
        xgb = XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=100,
                            objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
        xgb.fit(X_train , Y_train)

        ## Run model with only training data ##
        logging.warn('Test prediction accuracy')
        p_pred = xgb.predict(X_2)
        p_pred_i = le.inverse_transform(p_pred)
        p_pred_p = xgb.predict_proba(X_2)
        logging.warn('Accuracy: '+str(np.mean(p_pred_i == np.array(Y_test).ravel())))
        logging.warn('\n'+classification_report(p_pred_i,np.array(Y_test)))
        logging.warn('Log Loss: {}'.format(log_loss(np.array(Y_test).ravel(), p_pred_p)))
        logging.warn('Label Ranking Precision score: {}'.format(label_ranking_average_precision_score(cat_tst_lb, p_pred_p)))
        logging.warn('Label Ranking loss: {}'.format(label_ranking_loss(cat_tst_lb, p_pred_p)))

    ## Run model with all data and save ##
    if save_results:
        ''' Write results to a csv file
            NOTE: sorting is not done here
        '''
        logging.warn('Make predictions for final test set')
        logging.warn('Running model with all training data')
        xgb = XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=100,
                            objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
        xgb.fit(X , Y)
        X = np.concatenate((p.transform(final_X_test[CAT_COLS]).todense() \
                                ,im2.transform(np.array(final_X_test[NUM_COLS]))),axis=1)
        f_pred = xgb.predict_proba(X)

        ## Write to submissing file ##
        f_pred_df = pd.DataFrame(f_pred,columns=sorted(set(np.array(Y_train).ravel())))
        f_pred_df.index = np.array(final_X_test['id'])

        s = f_pred_df.stack()
        s2 = s.reset_index(level=0).reset_index(level=0)
        s2.columns = ['country','id','score']
        r = s2.groupby(['id'])['score'].rank(ascending=False)
        s3 = s2[r<=5]

        logging.warn('Writing to submission file')
        s3[['id','country','score']].to_csv('Data/submission.csv',index=False)

def run():
    declare_args(); load_data()
    user_features(update_columns=True)
    attach_age_buckets(update_columns=True)
    attach_sessions(collapse=False, pca=True, lm=True, update_columns=True, pca_n=20)
    component_isolation(categorical=True, numeric=True, update_columns=True)
    final_model(test=True, grid_cv=False, save_results=True)

# if __name__=='__main__':
#     run()
