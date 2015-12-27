
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

# Sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.discriminant_analysis     import LinearDiscriminantAnalysis as LDA            , QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,Imputer
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.linear_model import LinearRegression

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
TRAIN_DATA_FILE = 'Data/train_users.csv'

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


# ## Read data
# 

# In[4]:

## Read user data ## 
train_full = pd.read_csv(TRAIN_DATA_FILE).sort_values('id')
train_set, train_target = train_full[TEST_N:][USER_COLUMNS+TARGET_COLUMN],    train_full[TEST_N:][TARGET_COLUMN]
test_set, test_target = train_full[:TEST_N][USER_COLUMNS+TARGET_COLUMN],    train_full[:TEST_N][TARGET_COLUMN]


# In[5]:

## Read in data to predict for submission ##
final_test_set = pd.read_csv(TEST_DATA_FINAL_FILE)


# In[6]:

## Read supplemental datasets ## 
countries = pd.read_csv(COUNTRIES_FILE)
age_buckets = pd.read_csv(AGE_GENDER_BUCKETS_FILE)


# In[7]:

## Read session data ##
sessions = pd.read_csv(SESSIONS_FILE)


# #### Sessions

# #### User data

# In[8]:

train_set.shape


# In[9]:

train_set.head()


# In[10]:

train_set.index = train_set['id']
test_set.index = test_set['id']
final_test_set.index = final_test_set['id']


# In[11]:

train_set['gender'].value_counts()


# In[12]:

train_set.head()


# In[13]:

train_set.loc[train_set['age']>115,['age']] = np.nan
test_set.loc[test_set['age']>115,['age']] = np.nan
final_test_set.loc[final_test_set['age']>115,['age']] = np.nan


# In[14]:

train_set['date_created'] = pd.to_datetime(train_set['date_account_created'])
train_set['date_first_booking'] = pd.to_datetime(train_set['date_first_booking'])
train_set['year_created'] = train_set['date_created'].dt.year
train_set['month_created'] = train_set['date_created'].dt.month
train_set['year_first_booking'] = train_set['date_first_booking'].dt.year
train_set['month_first_booking'] = train_set['date_first_booking'].dt.month
train_set['days_to_first_booking'] = train_set['date_first_booking']-train_set['date_created']

# repeat with test 
test_set['date_created'] = pd.to_datetime(test_set['date_account_created'])
test_set['date_first_booking'] = pd.to_datetime(test_set['date_first_booking'])
test_set['year_created'] = test_set['date_created'].dt.year
test_set['month_created'] = test_set['date_created'].dt.month
test_set['year_first_booking'] = test_set['date_first_booking'].dt.year
test_set['month_first_booking'] = test_set['date_first_booking'].dt.month
test_set['days_to_first_booking'] = test_set['date_first_booking']-test_set['date_created']

# repeat with final test
final_test_set['date_created'] = pd.to_datetime(final_test_set['date_account_created'])
final_test_set['date_first_booking'] = pd.to_datetime(final_test_set['date_first_booking'])
final_test_set['year_created'] = final_test_set['date_created'].dt.year
final_test_set['month_created'] = final_test_set['date_created'].dt.month
final_test_set['year_first_booking'] = final_test_set['date_first_booking'].dt.year
final_test_set['month_first_booking'] = final_test_set['date_first_booking'].dt.month
final_test_set['days_to_first_booking'] = final_test_set['date_first_booking']-test_set['date_created']


# In[15]:

train_set.head()


# In[16]:

train_set['days_to_first_booking'].value_counts()


# In[17]:

train_set.loc[train_set['days_to_first_booking']<pd.Timedelta(0)              ,['days_to_first_booking']] = np.nan
train_set['days_to_first_booking'] =                 train_set['days_to_first_booking'].astype('timedelta64[D]')

test_set.loc[test_set['days_to_first_booking']<pd.Timedelta(0)              ,['days_to_first_booking']] = np.nan
test_set['days_to_first_booking'] =                 test_set['days_to_first_booking'].astype('timedelta64[D]')

final_test_set.loc[final_test_set['days_to_first_booking']<pd.Timedelta(0)              ,['days_to_first_booking']] = np.nan
final_test_set['days_to_first_booking'] =                 final_test_set['days_to_first_booking'].astype('timedelta64[D]')


# #### age buckets

# In[18]:

age_buckets.head()


# In[19]:

age_buckets['age_merge'] = (np.floor(                  np.array([int(re.split(r'[-+]',str(x))[0])                   for x in age_buckets['age_bucket']]            )/10)*10).astype('int')


# In[20]:

age_buckets.index = age_buckets['age_merge'].astype('string')             +'-'+age_buckets['country_destination']             +'-'+age_buckets['gender'].str.lower()


# In[21]:

for c in set(countries['country_destination']):
    train_set['age_merge'+'-'+c] = (
                        np.floor(\
                            train_set['age']/10)*10\
                        )\
                            .fillna(0)\
                            .astype('int')\
                            .astype('string') \
                        +'-'+c \
                        +'-'+train_set['gender'].str.lower()
    test_set['age_merge'+'-'+c] = (
                        np.floor(\
                            test_set['age']/10)*10\
                        )\
                            .fillna(0)\
                            .astype('int')\
                            .astype('string') \
                        +'-'+c \
                        +'-'+test_set['gender'].str.lower()
    final_test_set['age_merge'+'-'+c] = (
                        np.floor(\
                            final_test_set['age']/10)*10\
                        )\
                            .fillna(0)\
                            .astype('int')\
                            .astype('string') \
                        +'-'+c \
                        +'-'+test_set['gender'].str.lower()


# In[22]:

age_buckets = age_buckets[[
        'age_merge' \
        ,'country_destination' \
        ,'gender' \
        ,'population_in_thousands']] \
    .groupby(['age_merge','country_destination','gender']).sum()


# In[23]:

age_buckets.index = pd.Series([ str(i[0])+'-'+i[1]+'-'+i[2] for i in age_buckets.index])


# In[24]:

for c in set(countries['country_destination']):
    train_set = pd.merge(
        train_set \
         , age_buckets \
         , left_on=['age_merge'+'-'+c] \
         , right_index=True \
         , how='outer' \
         , suffixes=(c,c)
    )
    test_set = pd.merge(
        test_set \
         , age_buckets \
         , left_on=['age_merge'+'-'+c] \
         , right_index=True \
         , how='outer' \
         , suffixes=(c,c)
    )
    final_test_set = pd.merge(
        final_test_set \
         , age_buckets \
         , left_on=['age_merge'+'-'+c] \
         , right_index=True \
         , how='left' \
         , suffixes=(c,c)
    )
print(train_set.shape)


# In[25]:

train_set = train_set.drop_duplicates(['id'])
test_set = test_set.drop_duplicates(['id'])


# In[26]:

train_set['population_estimate'] = 0
test_set['population_estimate'] = 0
final_test_set['population_estimate'] = 0
for c in set(countries['country_destination']):
    try:
        train_set.loc[:,'population_estimate'] =             train_set.loc[:,'population_estimate']            +np.nansum(train_set.loc[:,'population_in_thousands'+c]                       ,axis=1)
        test_set.loc[:,'population_estimate'] =             test_set.loc[:,'population_estimate']            +np.nansum(test_set.loc[:,'population_in_thousands'+c]                       ,axis=1)
        final_test_set.loc[:,'population_estimate'] =             final_test_set.loc[:,'population_estimate']            +np.nansum(final_test_set.loc[:,'population_in_thousands'+c]                       ,axis=1)
    except KeyError:
        pass


# #### Add country features

# In[27]:

countries


# In[28]:

set(train_set['language'])


# #### Add session data

# In[29]:

sessions.shape


# In[30]:

sessions.head()


# In[31]:

cf = ['action','action_type','action_detail','device_type']
s = sessions[cf].copy().fillna('missing')
mcl = MultiColumnLabelEncoder()
ohe = OneHotEncoder()
x = ohe.fit_transform(
    mcl.fit_transform(s)
).todense()


# In[32]:

try:
    sessions_new = pd.read_csv('Data/sessions_new.csv',index_col=0)
except IOError:
    sessions_new = []


# In[33]:

run = True if len(sessions_new)==0 else False


# In[34]:

if run:
    n = 100
    loops = sessions.shape[0]//n*np.arange(n)

    o = []
    start_time = DT.datetime.now()
    for i,l in enumerate(loops):
        try:
            a,b = loops[i],loops[i+1]
        except:
            a,b = loops[i],sessions.shape[0]

        sessions_new = pd.DataFrame(np.concatenate(            (
                sessions[['user_id']][a:b]\
                , x[a:b]\
                , sessions[['secs_elapsed']][a:b]
            )
            , axis=1
        ))
        sessions_grouped = sessions_new.groupby([0]).sum()
        o.append(sessions_grouped)

        if i%10==0:
            this_time = DT.datetime.now()
            time_change = (this_time - start_time).seconds
            per_second = b*1.0 / (time_change)
            total_time = sessions.shape[0] / per_second / 60
            pct = b*1.0 / sessions.shape[0]
            print('finished {}%, {} mins est. time remaining'                    .format(round(pct*100,2)                            ,(1-pct)*total_time))
    sessions_new = pd.concat(o,ignore_index=True)
    sessions_new['user_id'] = pd.concat(o).index
    sessions_new = sessions_new.groupby('user_id').sum()
    sessions_new.to_csv('Data/sessions_new.csv',index=True,header=True)


# In[35]:

target = pd.DataFrame({'country_destination':train_set['country_destination']})
target.index = train_set['id']
merged = pd.merge(            sessions_new            , target            , how='inner'            , left_index=True
            , right_index=True
         )
merged.head()


# In[36]:

merged.describe()


# In[37]:

np.max(np.array(merged)[:,:-1])


# In[38]:

ss = StandardScaler(with_mean=False)
ii = Imputer(strategy='most_frequent')
lda = LDA()
p = Pipeline([('ii',ii),('ss', ss)])
merged_new = p.fit_transform(np.array(merged)[:,:-1])
l = lda.fit_transform(merged_new, np.array(merged)[:,-1:])


# In[39]:

sessions_lda = pd.DataFrame(l                            ,index=merged.index                            ,columns=[ 'lda_'+str(i)                                       for i in range(l.shape[1]) ])


# In[40]:

sessions_lda.head()


# In[41]:

pca = PCA(n_components=1)
p = pca.fit_transform(sessions_lda)


# In[42]:

t = pd.merge(            train_set            , pd.DataFrame({'lda':p[:,0]},index=sessions_lda.index)            , how='inner'            , left_index=True
            , right_index=True
    )


# In[48]:

cols = [
 'gender',
 'age',
 #'signup_method',
 'signup_flow',
 'language',
 'affiliate_channel',
 'affiliate_provider',
 'first_affiliate_tracked',
 'signup_app',
 'first_device_type',
 'first_browser',
 'year_created' ,
 'month_created' ,
 'year_first_booking' ,
 'month_first_booking' ,
 'days_to_first_booking',
 'population_estimate'
]
t[cols+['lda']].head()


# In[49]:

lm = LinearRegression()
mcl = MultiColumnLabelEncoder()
mm = MinMaxScaler()
ohe = OneHotEncoder()
ss = StandardScaler(with_mean=False)
ii = Imputer(strategy='most_frequent')
p = Pipeline([('mcl',mcl),('ii',ii),('mm',mm),('ss', ss)])
lda = p.fit_transform(t[cols])
lda_lm = lm.fit(lda,t['lda'])


# #### Compile dataset

# In[45]:

cat_cols = [
    'gender',
    #'signup_method',
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
num_cols = [
    'age',
    'days_to_first_booking',
    'population_estimate' 
]
print(train_set[cat_cols + num_cols].shape)


# In[46]:

train_set[cat_cols+num_cols].head()


# In[50]:

tr_lda = p.fit_transform(train_set[cat_cols+num_cols])
tst_lda = p.transform(test_set[cat_cols+num_cols])
final_tst_lda = p.transform(final_test_set[cat_cols+num_cols])
lda = lda_lm.predict(tr_lda)
tst_lda = lda_lm.predict(tst_lda)
final_tst_lda = lda_lm.predict(final_tst_lda)
#train_set['session_lda'] = lda
#test_set['session_lda'] = tst_lda
#final_test_set['session_lda'] = final_tst_lda


# In[51]:

mcl = MultiColumnLabelEncoder()
mm = MinMaxScaler()
ohe = OneHotEncoder()
ss = StandardScaler(with_mean=False)
ii = Imputer(strategy='most_frequent')
ii2 = Imputer(strategy='mean')
p = Pipeline([('mcl',mcl),('ii',ii),('ohe',ohe)])
p2 = Pipeline([('ii',ii2)]) #,('ss',ss),('mm',mm)])


# In[52]:

z = p.fit_transform(train_set[cat_cols])
zB = p2.fit_transform(train_set[num_cols]) #+['session_lda']])
z2 = p.transform(test_set[cat_cols])
z2B = p2.transform(test_set[num_cols]) #+['session_lda']])
z3 = p.transform(final_test_set[cat_cols])
z3B = p2.transform(final_test_set[num_cols]) #+['session_lda']])

train_set_new = np.concatenate((z.todense(),zB),axis=1)
test_set_new = np.concatenate((z2.todense(),z2B),axis=1)
final_test_set_new = np.concatenate((z3.todense(),z3B),axis=1)


# In[53]:

print(train_set_new[:,:].shape)


# In[54]:

train_target = train_set['country_destination'].fillna('unknown')
test_target = test_set['country_destination'].fillna('unknown')
print(train_target.shape)


# In[55]:

lda = LDA(n_components=6)
l = lda.fit_transform(train_set_new, np.array(train_target))
l_tst = lda.transform(test_set_new)
final_l_tst = lda.transform(final_test_set_new)

lb = LabelBinarizer()
cat = lb.fit_transform(np.array(train_target))
cat_tst = lb.transform(np.array(test_target))

le = LabelEncoder()
cat_le = le.fit_transform(np.array(train_target))
cat_tst_le = le.transform(np.array(test_target))


'''
n = l.shape[1]
nn = Sequential()
nn.add(GaussianDropout(0.3))
nn.add(Dense(input_dim=n, output_dim=32, init="uniform"))
nn.add(Activation("tanh"))
nn.add(PReLU((32,)))
nn.add(Dropout(0.3))
nn.add(Dense(input_dim=32, output_dim=13, init="glorot_uniform"))
nn.add(Activation("softmax"))
nn.compile(loss='categorical_crossentropy', optimizer='sgd')
nn.fit(l, cat, nb_epoch=10)

p_pred = nn.predict(l_tst)
p_pred_i = lb.inverse_transform(p_pred)
'''

xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)      
xgb.fit(l, cat_le)

p_pred = xgb.predict(l_tst)
p_pred_i = le.inverse_transform(p_pred)

print(np.mean(p_pred_i == np.array(test_target)))
print(classification_report(p_pred_i,np.array(test_target)))

f_pred = xgb.predict_proba(final_l_tst)

f_pred_df = pd.DataFrame(f_pred,columns=sorted(set(train_target)))
f_pred_df.index = np.array(final_test_set['id'])

s = f_pred_df.stack()
s2 = s.reset_index(level=0).reset_index(level=0)
s2.columns = ['country','id','score']
r = s2.groupby(['id'])['score'].rank(ascending=False)
s3 = s2[r<=5]
s3[['id','country','score']].to_csv('Data/submission.csv',index=False)

