
# coding: utf-8

# AirBnB recruiting kaggle
# ------
#
# https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

# ## Load libraries

# In[1]:

# Core
from __future__ import print_function
import datetime as DT
import pandas as pd
import numpy as np
import re
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

# Sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,Imputer
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import AdaBoostClassifier

# Metrics
from sklearn.metrics import log_loss, classification_report

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

# In[2]:

class MultiColumnLabelEncoder:
    ''' Create a class that encodes
        labels for a matrix of data
    '''
    def __init__(self, columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def get_params(self, deep=True):
        out = dict()
        if self.columns: out['columns'] = columns
        return out

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder().
        '''
        numerics = [np.float16, np.float32, np.float64]
        ints = [np.int16, np.int32, np.int64]
        output = X.copy()

        for colname,col in output.iteritems():
            if col.dtype not in numerics+ints:
                # Turn text columns into ints
                output[colname] = LabelEncoder().fit_transform(output[colname])
            elif col.dtype in numerics:
                # handle floats with scaling
                # output[colname] = scale(output[colname])
                pass
            elif col.dtype in ints:
                pass # leave integers alone
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# ## Declare Args

# In[3]:

## Files ##
AGE_GENDER_BUCKETS_FILE = 'Data/age_gender_bkts.csv'
COUNTRIES_FILE = 'Data/countries.csv'
SAMPLE_SUBMISSION_FILE = 'Data/sample_submission.csv'
SESSIONS_FILE = 'Data/sessions.csv'
TEST_DATA_FINAL_FILE = 'Data/test_users.csv'
TRAIN_DATA_FILE = 'Data/train_users_2.csv'

## Model args ##
TEST_N = 20000

## Fields ##
USER_COLUMNS = ['id',
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
 'first_browser']
TARGET_COLUMN = ['country_destination']

SESSION_COLUMNS = ['user_id',
 'action',
 'action_type',
 'action_detail',
 'device_type',
 'secs_elapsed']

AGE_BUCKET_COLUMNS = ['age_bucket',
 'country_destination',
 'gender',
 'population_in_thousands',
 'year']

CAT_COLS = [
    'gender',
    'signup_method', # New method weibo in final test data
    'signup_flow',
    'language',
    'affiliate_channel',
    'affiliate_provider',
    'first_affiliate_tracked',
    'signup_app',
    'first_device_type',
    'first_browser',
    'year_created',
    'month_created',
    'year_first_booking' ,
    'month_first_booking' ,
]
NUM_COLS = [
    'age',
    'days_to_first_booking'
]

# ## Read data
#

logging.warn('Loading data files')

## Read user data ##
train_full = pd.read_csv(TRAIN_DATA_FILE).sort_values('id')
train_full = train_full.iloc[np.random.permutation(len(train_full))]
train_set, train_target = train_full[TEST_N:][USER_COLUMNS+TARGET_COLUMN] \
                                            , train_full[TEST_N:][TARGET_COLUMN]
test_set, test_target = train_full[:TEST_N][USER_COLUMNS+TARGET_COLUMN] \
                                            , train_full[:TEST_N][TARGET_COLUMN]

## Read in data to predict for submission ##
final_test_set = pd.read_csv(TEST_DATA_FINAL_FILE)

## Read supplemental datasets ##
countries = pd.read_csv(COUNTRIES_FILE)
age_buckets = pd.read_csv(AGE_GENDER_BUCKETS_FILE)

## Read session data ##
sessions = pd.read_csv(SESSIONS_FILE)

# #### User data
logging.warn('Processing user data')

train_set.index = train_set['id']
test_set.index = test_set['id']
final_test_set.index = final_test_set['id']

train_set.loc[train_set['age']>115,['age']] = np.nan
test_set.loc[test_set['age']>115,['age']] = np.nan
final_test_set.loc[final_test_set['age']>115,['age']] = np.nan

## add new date features ##
train_set['date_created'] = pd.to_datetime(train_set['date_account_created'])
train_set['date_first_booking'] = pd.to_datetime(train_set['date_first_booking'])
train_set['year_created'] = train_set['date_created'].dt.year
train_set['month_created'] = train_set['date_created'].dt.month
train_set['year_first_booking'] = train_set['date_first_booking'].dt.year
train_set['month_first_booking'] = train_set['date_first_booking'].dt.month
train_set['days_to_first_booking'] = train_set['date_first_booking']-train_set['date_created']

## repeat with test ##
test_set['date_created'] = pd.to_datetime(test_set['date_account_created'])
test_set['date_first_booking'] = pd.to_datetime(test_set['date_first_booking'])
test_set['year_created'] = test_set['date_created'].dt.year
test_set['month_created'] = test_set['date_created'].dt.month
test_set['year_first_booking'] = test_set['date_first_booking'].dt.year
test_set['month_first_booking'] = test_set['date_first_booking'].dt.month
test_set['days_to_first_booking'] = test_set['date_first_booking']-test_set['date_created']

## repeat with final test ##
final_test_set['date_created'] = pd.to_datetime(final_test_set['date_account_created'])
final_test_set['date_first_booking'] = pd.to_datetime(final_test_set['date_first_booking'])
final_test_set['year_created'] = final_test_set['date_created'].dt.year
final_test_set['month_created'] = final_test_set['date_created'].dt.month
final_test_set['year_first_booking'] = final_test_set['date_first_booking'].dt.year
final_test_set['month_first_booking'] = final_test_set['date_first_booking'].dt.month
final_test_set['days_to_first_booking'] = final_test_set['date_first_booking']-test_set['date_created']

## add new date features ##
train_set.loc[train_set['days_to_first_booking']<pd.Timedelta(0) \
                    ,['days_to_first_booking']] = np.nan
train_set['days_to_first_booking'] = \
                    train_set['days_to_first_booking'].astype('timedelta64[D]')

test_set.loc[test_set['days_to_first_booking']<pd.Timedelta(0)\
                    ,['days_to_first_booking']] = np.nan
test_set['days_to_first_booking'] = \
                    test_set['days_to_first_booking'].astype('timedelta64[D]')

final_test_set.loc[final_test_set['days_to_first_booking']<pd.Timedelta(0) \
                    ,['days_to_first_booking']] = np.nan
final_test_set['days_to_first_booking'] = \
                    final_test_set['days_to_first_booking'].astype('timedelta64[D]')

## Choose random record in training data to assign 'weibo' signup method ##
## This avoids issues later with one hot encoding ##
train_set.loc[train_set.iloc[22]['id'],['signup_method']] = 'weibo'

# #### Sessions
logging.warn('Processing session data model')
cf = ['action','action_type','action_detail','device_type']
s = sessions[cf].copy().fillna('missing')
mcl = MultiColumnLabelEncoder()
ohe = OneHotEncoder()
x = ohe.fit_transform(
    mcl.fit_transform(s)
)
n = x.shape[1]//50

def minimize_df(i,lim,n,nparr):
    m = min(lim,(i+1)*n)
    return pd.DataFrame(nparr[:,i*n:m].toarray() \
                ,index=sessions['user_id']) \
                .groupby(level=0).sum()

z = ( minimize_df(y,x.shape[1],50,x) for y in range(n+1) )

sessions_new = pd.concat(z,axis=1)
sessions_new.columns = [ 'session_'+str(i) for i in range(len(sessions_new.columns)) ]

train_set = pd.merge(train_set, sessions_new, how='left', left_index=True, right_index=True)
test_set = pd.merge(test_set, sessions_new, how='left', left_index=True, right_index=True)
final_test_set = pd.merge(final_test_set, sessions_new, how='left', left_index=True, right_index=True)

train_set.loc[:,sessions_new.columns] = train_set.loc[:,sessions_new.columns].fillna(0)
test_set.loc[:,sessions_new.columns] = test_set.loc[:,sessions_new.columns].fillna(0)
final_test_set.loc[:,sessions_new.columns] = final_test_set.loc[:,sessions_new.columns].fillna(0)

c = 5
pca = PCA(n_components=c)
tr_pca = pd.DataFrame( pca.fit_transform(train_set.loc[:,sessions_new.columns]) \
                    , columns = ['pca_session_' + str(i) for i in range(c)] \
                    , index = train_set.index \
                )
tst_pca = pd.DataFrame( pca.transform(test_set.loc[:,sessions_new.columns]) \
                    , columns = ['pca_session_' + str(i) for i in range(c)]
                    , index = test_set.index \
                )
fnl_pca = pd.DataFrame( pca.transform(final_test_set.loc[:,sessions_new.columns]) \
                    , columns = ['pca_session_' + str(i) for i in range(c)]
                    , index = final_test_set.index \
                )

train_set = pd.concat([train_set,tr_pca],axis=1)
test_set = pd.concat([test_set,tst_pca],axis=1)
final_test_set = pd.concat([final_test_set,fnl_pca],axis=1)

# NUM_COLS += list( sessions_new.columns )
NUM_COLS += ['pca_session_' + str(i) for i in range(c)]

logging.warn('Create boosted trees model with training data')
## Encode categories ##
le = LabelEncoder()
cat_le = le.fit_transform(np.array(train_target).ravel())
cat_tst_le = le.transform(np.array(test_target).ravel())

mcl = MultiColumnLabelEncoder() ; ohe = OneHotEncoder() ; im = Imputer(strategy='most_frequent')
im2 = Imputer(strategy='mean')
p = Pipeline([('mcl',mcl),('im',im),('ohe',ohe)])
'''
params_grid = {'learning_rate':[0.3,0.1,0.05,0.02,0.01]
		, 'max_depth':[ 4, 6 ]}

xgb = XGBClassifier(n_estimators=50, objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
gs_csv = GridSearchCV(xgb, params_grid).fit(train_set_new, cat_le)
print(gs_csv.best_params_)
'''

## Set up X,Y data for modeling ##
X_1 = np.concatenate((p.fit_transform(train_set[CAT_COLS]).todense() \
                        ,im2.fit_transform(np.array(train_set[NUM_COLS]))),axis=1)
X_2 = np.concatenate((p.transform(test_set[CAT_COLS]).todense() \
                        ,im2.transform(np.array(test_set[NUM_COLS]))),axis=1)
Y = cat_le

## Get rid of unimportant ##
logging.warn('Extracting meaningful features')
abc = AdaBoostClassifier(learning_rate=0.01)
abc.fit( X_1 , Y  )
fi = abc.feature_importances_
features = np.argsort(fi)[::-1][:25]

## Run model with only training data ##
logging.warn('Running model with training data')
xgb = XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=50,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
xgb.fit(X_1 , Y)

## Run model with only training data ##
logging.warn('Test prediction accuracy')
p_pred = xgb.predict(X_2)
p_pred_i = le.inverse_transform(p_pred)
p_pred_p = xgb.predict_proba(X_2)
logging.warn('Accuracy: '+str(np.mean(p_pred_i == np.array(test_target).ravel())))
logging.warn('\n'+classification_report(p_pred_i,np.array(test_target)))
logging.warn('Log Loss: {}'.format(log_loss(np.array(test_target).ravel(), p_pred_p)))

## Run model with all data ##
logging.warn('Re-running model with all training data')
xgb = XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=50,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
X = np.concatenate((X_1,X_2),axis=1)
Y = np.concatenate([cat_le,cat_tst_le])
xgb.fit(X , Y)

logging.warn('Make predictions for final test set')
X = np.concatenate((p.transform(final_test_set[CAT_COLS]).todense() \
                        ,im2.transform(np.array(final_test_set[NUM_COLS]))),axis=1)
f_pred = xgb.predict_proba(X)

## Write to submissing file ##
f_pred_df = pd.DataFrame(f_pred,columns=sorted(set(np.array(train_target).ravel())))
f_pred_df.index = np.array(final_test_set['id'])

s = f_pred_df.stack()
s2 = s.reset_index(level=0).reset_index(level=0)
s2.columns = ['country','id','score']
r = s2.groupby(['id'])['score'].rank(ascending=False)
s3 = s2[r<=5]

logging.warn('Writing to submission file')
s3[['id','country','score']].to_csv('Data/submission.csv',index=False)
