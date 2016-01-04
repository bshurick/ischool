
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
from sklearn.linear_model import LinearRegression
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
    'signup_method',
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
    'days_to_first_booking',
    'population_estimate'
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

target = pd.DataFrame({'country_destination':train_set['country_destination']})
target.index = train_set['id']
merged = pd.merge(\
            sessions_new\
            , target\
            , how='inner'\
            , left_index=True
            , right_index=True
         )

## Extract most importance features ##
abc = AdaBoostClassifier(learning_rate=0.1)
abc.fit( np.array(merged)[:,:-1] , np.array(merged)[:,-1:].ravel() )
fi = abc.feature_importances_
features = np.argsort(fi)[::-1][:20]

## Collapse into smaller feature set ##
pca = PCA(n_components=4)
pca_features = pd.DataFrame(pca.fit_transform(np.array(sessions_new)[:,features])\
                            , index = sessions_new.index)
logging.warn('Session PCA explained variance '+str(np.sum(pca.explained_variance_ratio_)))

## Create prediction model for features ##
tr_cat = train_set.loc[:,CAT_COLS]
tr_cat.index = train_set['id']
tr_num = train_set.loc[:,NUM_COLS]
tr_num.index = train_set['id']

tst_cat = test_set.loc[:,CAT_COLS]
tst_cat.index = test_set['id']
tst_num = test_set.loc[:,NUM_COLS]
tst_num.index = test_set['id']

merged_cats = pd.merge(tr_cat \
                        , pca_features \
                        , how='inner' \
                        , left_index=True \
                        , right_index=True  )
merged_nums = pd.merge(tr_num \
                        , pca_features \
                        , how='inner' \
                        , left_index=True \
                        , right_index=True  )
mcl = MultiColumnLabelEncoder()
mm = MinMaxScaler()
ohe = OneHotEncoder()
ss = StandardScaler(with_mean=False)
ii = Imputer(strategy='most_frequent')
ii2 = Imputer(strategy='mean')
lms = [ LinearRegression() for l in np.arange(pca_features.shape[1]) ]
p1 = Pipeline([('mcl',mcl),('ii',ii),('ohe',ohe)])
p2 = Pipeline([('ii',ii2),('ss',ss),('mm',mm)])

trcat_transformed = p1.fit_transform(tr_cat).todense()
trnum_transformed = p2.fit_transform(tr_num)
trcombined = np.concatenate((trcat_transformed, trnum_transformed), axis=1)

tstcat_transformed = p1.fit_transform(tst_cat).todense()
tstnum_transformed = p2.fit_transform(tst_num)
tstcombined = np.concatenate((tstcat_transformed, tstnum_transformed), axis=1)

mcat_transformed = p1.transform(merged_cats.iloc[:,:-4]).todense()
mnum_transformed = p2.transform(merged_nums.iloc[:,:-4])
mcombined = np.concatenate((mcat_transformed, mnum_transformed), axis=1)

for i,lm in enumerate(lms):
    lm.fit(mcombined, merged_cats.iloc[:,merged_cats.shape[1]-i-1])
    train_set.loc[:,'pca_'+str(i)] = lm.predict(trcombined)
    test_set.loc[:,'pca_'+str(i)] = lm.predict(trcombined)
    lms[i] = lm


# #### User data
logging.warn('Processing user data')

train_set.index = train_set['id']
test_set.index = test_set['id']
final_test_set.index = final_test_set['id']

train_set.loc[train_set['age']>115,['age']] = np.nan
test_set.loc[test_set['age']>115,['age']] = np.nan
final_test_set.loc[final_test_set['age']>115,['age']] = np.nan

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

train_set.loc[train_set['days_to_first_booking']<pd.Timedelta(0)              ,['days_to_first_booking']] = np.nan
train_set['days_to_first_booking'] =                 train_set['days_to_first_booking'].astype('timedelta64[D]')

test_set.loc[test_set['days_to_first_booking']<pd.Timedelta(0)              ,['days_to_first_booking']] = np.nan
test_set['days_to_first_booking'] =                 test_set['days_to_first_booking'].astype('timedelta64[D]')

final_test_set.loc[final_test_set['days_to_first_booking']<pd.Timedelta(0)              ,['days_to_first_booking']] = np.nan
final_test_set['days_to_first_booking'] =                 final_test_set['days_to_first_booking'].astype('timedelta64[D]')


# #### age buckets

# logging.warn('Processing age bucket data')
# age_buckets['age_merge'] = (np.floor( \
#     np.array([int(re.split(r'[-+]',str(x))[0]) \
#         for x in age_buckets['age_bucket']] \
#     )/10)*10).astype('int')
#
# age_buckets.index = age_buckets['age_merge'].astype('string') \
#                                     +'-'+age_buckets['country_destination'] \
#                                     +'-'+age_buckets['gender'].str.lower()
#
# for c in set(countries['country_destination']):
#     train_set['age_merge'+'-'+c] = (
#                         np.floor(\
#                             train_set['age']/10)*10\
#                         )\
#                             .fillna(0)\
#                             .astype('int')\
#                             .astype('string') \
#                         +'-'+c \
#                         +'-'+train_set['gender'].str.lower()
#     test_set['age_merge'+'-'+c] = (
#                         np.floor(\
#                             test_set['age']/10)*10\
#                         )\
#                             .fillna(0)\
#                             .astype('int')\
#                             .astype('string') \
#                         +'-'+c \
#                         +'-'+test_set['gender'].str.lower()
#     final_test_set['age_merge'+'-'+c] = (
#                         np.floor(\
#                             final_test_set['age']/10)*10\
#                         )\
#                             .fillna(0)\
#                             .astype('int')\
#                             .astype('string') \
#                         +'-'+c \
#                         +'-'+test_set['gender'].str.lower()
#
# age_buckets = age_buckets[[
#         'age_merge' \
#         ,'country_destination' \
#         ,'gender' \
#         ,'population_in_thousands']] \
#     .groupby(['age_merge','country_destination','gender']).sum()
#
# age_buckets.index = pd.Series([ str(i[0])+'-'+i[1]+'-'+i[2] for i in age_buckets.index])
#
# for c in set(countries['country_destination']):
#     train_set = pd.merge(
#         train_set \
#          , age_buckets \
#          , left_on=['age_merge'+'-'+c] \
#          , right_index=True \
#          , how='outer' \
#          , suffixes=(c,c)
#     )
#     test_set = pd.merge(
#         test_set \
#          , age_buckets \
#          , left_on=['age_merge'+'-'+c] \
#          , right_index=True \
#          , how='outer' \
#          , suffixes=(c,c)
#     )
#     final_test_set = pd.merge(
#         final_test_set \
#          , age_buckets \
#          , left_on=['age_merge'+'-'+c] \
#          , right_index=True \
#          , how='left' \
#          , suffixes=(c,c)
#     )
#
# train_set = train_set.drop_duplicates(['id'])
# test_set = test_set.drop_duplicates(['id'])
#
#
# train_target = train_set['country_destination'].fillna('unknown')
# test_target = test_set['country_destination'].fillna('unknown')
# print(train_target.shape)

# s_sort = sorted(enumerate(s),key=lambda x: x[1],reverse=True)
# features = [i[0] for i in s_sort if i[1]>=min(sorted(s,reverse=True)[:30])]

le = LabelEncoder()
cat_le = le.fit_transform(np.array(train_target))
cat_tst_le = le.transform(np.array(test_target))

'''
params_grid = {'learning_rate':[0.3,0.1,0.05,0.02,0.01]
		, 'max_depth':[ 4, 6 ]}

xgb = XGBClassifier(n_estimators=50, objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
gs_csv = GridSearchCV(xgb, params_grid).fit(train_set_new, cat_le)
print(gs_csv.best_params_)
'''

xgb = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=50,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
xgb.fit(np.concatenate([train_set,test_set]), np.concatenate([cat_le,cat_tst_le]))
p_pred = xgb.predict(test_set_new)
p_pred_i = le.inverse_transform(p_pred)

print(np.mean(p_pred_i == np.array(test_target)))
print(classification_report(p_pred_i,np.array(test_target)))

f_pred = xgb.predict_proba(final_test_set_new)

f_pred_df = pd.DataFrame(f_pred,columns=sorted(set(train_target)))
f_pred_df.index = np.array(final_test_set['id'])

s = f_pred_df.stack()
s2 = s.reset_index(level=0).reset_index(level=0)
s2.columns = ['country','id','score']
r = s2.groupby(['id'])['score'].rank(ascending=False)
s3 = s2[r<=5]
s3[['id','country','score']].to_csv('Data/submission.csv',index=False)
