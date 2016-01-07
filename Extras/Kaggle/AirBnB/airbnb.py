
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
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, ElasticNetCV
from sklearn.ensemble import AdaBoostClassifier

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
np.random.seed(9)
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

## Prepare data for regression modeling ##
target = pd.DataFrame({'country_destination':train_set['country_destination']})
target.index = train_set['id']
merged = pd.merge(\
            sessions_new\
            , target\
            , how='left'\
            , left_index=True
            , right_index=True
         )

## Extract most importance features ##
logging.warn('Extracting meaningful features')
abc = AdaBoostClassifier(learning_rate=0.1)
abc.fit( np.array(merged)[:,:-1] , np.array(merged)[:,-1:].ravel() )
fi = abc.feature_importances_
components = 50
features = np.argsort(fi)[::-1][:components]

# logging.warn('Collapsing feature set using PCA')
# pca = PCA(n_components=components)
# pca_features = pd.DataFrame(pca.fit_transform(np.array(sessions_new)[:,features])\
#                             , index = sessions_new.index)
# logging.warn('Session PCA explained variance '+str(np.sum(pca.explained_variance_ratio_)))
lm_features = sessions_new.iloc[:,features]
lm_features.columns = range(components)

## Create prediction model for features ##
logging.warn('Creating regression model for session features')

## Split out category and numeric columns ##
tr_cat = train_set.loc[:,CAT_COLS]
tr_cat.index = train_set['id']
tr_num = train_set.loc[:,NUM_COLS]
tr_num.index = train_set['id']

tst_cat = test_set.loc[:,CAT_COLS]
tst_cat.index = test_set['id']
tst_num = test_set.loc[:,NUM_COLS]
tst_num.index = test_set['id']

final_tst_cat = final_test_set.loc[:,CAT_COLS]
final_tst_cat.index = final_test_set['id']
final_tst_num = final_test_set.loc[:,NUM_COLS]
final_tst_num.index = final_test_set['id']

## Merge with new features ##
merged_cats = pd.merge(tr_cat \
                        , lm_features \
                        , how='inner' \
                        , left_index=True \
                        , right_index=True  )
merged_nums = pd.merge(tr_num \
                        , lm_features \
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

tstcat_transformed = p1.transform(tst_cat).todense()
tstnum_transformed = p2.transform(tst_num)
tstcombined = np.concatenate((tstcat_transformed, tstnum_transformed), axis=1)

tstcat_final_transformed = p1.transform(final_tst_cat).todense()
tstnum_final_transformed = p2.transform(final_tst_num)
tstcombined_final = np.concatenate((tstcat_final_transformed, tstnum_final_transformed), axis=1)

mcat_transformed = p1.transform(merged_cats.iloc[:,:-1*components]).todense()
mnum_transformed = p2.transform(merged_nums.iloc[:,:-1*components])
mcombined = np.concatenate((mcat_transformed, mnum_transformed), axis=1)

lm_cvs = [ ElasticNetCV( \
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1] \
            , alphas=[0.001,0.01,0.05,0.1,0.5,0.9] \
            , max_iter=1200, n_jobs=2
        ) \
        for l in np.arange(components) ]
for i,lm in enumerate(lm_cvs):
    lm.fit(mcombined, merged_cats.iloc[:,merged_cats.shape[1]-i-1])

for l in lm_cvs: logging.warn('L1: {} Alpha: {}'.format(l.l1_ratio_,l.alpha_))
lms = [ ElasticNet(l1_ratio=l.l1_ratio_, alpha=l.alpha_, normalize=True) \
            for l in lm_cvs ]

for i,lm in enumerate(lms):
    lm.fit(mcombined, merged_cats.iloc[:,merged_cats.shape[1]-i-1])
    train_set.loc[:,'lm_'+str(i)] = lm.predict(trcombined)
    test_set.loc[:,'lm_'+str(i)] = lm.predict(tstcombined)
    final_test_set.loc[:,'lm_'+str(i)] = lm.predict(tstcombined_final)
    lms[i] = lm

merged_tst = pd.merge(test_set \
                        , lm_features \
                        , how='inner' \
                        , left_index=True \
                        , right_index=True  )
for i in range(components):
    logging.warn('MSE {}: {}'.format(i \
        , np.sqrt(np.mean(np.sum((merged_tst['lm_'+str(i)] \
                                        - merged_tst[i])**2))) \
    ))
    # train_set.loc[merged_nums.index,'lm_'+str(i)] = merged_nums[i]
    # test_set.loc[merged_nums_tst.index,'lm_'+str(i)] = merged_nums_tst[i]

## PCA ##
logging.warn('Collapse session features with PCA')
c = 7
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
logging.warn('PCA Explained variance: {}'.format(np.sum(pca.explained_variance_ratio_)))

train_set = pd.concat([train_set,tr_pca],axis=1)
test_set = pd.concat([test_set,tst_pca],axis=1)
final_test_set = pd.concat([final_test_set,fnl_pca],axis=1)

NUM_COLS += list( sessions_new.columns )
NUM_COLS += ['pca_session_' + str(i) for i in range(c)]
NUM_COLS += [ 'lm_'+str(i) for i in range(components) ]

logging.warn('Create boosted trees model with training data')
## Encode categories ##
le = LabelEncoder()
cat_le = le.fit_transform(np.array(train_target).ravel())
cat_tst_le = le.transform(np.array(test_target).ravel())

lb = LabelBinarizer()
cat_lb = lb.fit_transform(np.array(train_target).ravel())
cat_tst_lb = lb.transform(np.array(test_target).ravel())

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

# Get rid of unimportant ##
# logging.warn('Extracting meaningful features')
# abc = AdaBoostClassifier(learning_rate=0.01)
# abc.fit( X_1 , Y  )
# fi = abc.feature_importances_
# features = np.argsort(fi)[::-1][:25]

## Run model with only training data ##
logging.warn('Running model with training data')
xgb = XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=100,
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
logging.warn('Label Ranking Precision score: {}'.format(label_ranking_average_precision_score(cat_tst_lb, p_pred_p)))
logging.warn('Label Ranking loss: {}'.format(label_ranking_loss(cat_tst_lb, p_pred_p)))

## Run model with all data ##
logging.warn('Re-running model with all training data')
xgb = XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=100,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
X = np.concatenate((X_1,X_2))
Y = np.concatenate((cat_le,cat_tst_le))
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
