
# coding: utf-8

# Final Project 
# ======
# 
# Kaggle Competition 
# -----
# 
# For this project I chose to do the active competition [San Francisco Crime Classification](https://www.kaggle.com/c/sf-crime/). I'll test out a number of different algorithms with test data. I am interested in this type of analysis as it is data science that contributes to the common good.

# ### Part 0: Setup

# In[1]:

# Pickle
import pickle

# Pandas and numpy
import pandas as pd
import numpy as np

# Python functions
from dateutil.relativedelta import *
from datetime import datetime
import re, math

# sklearn functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler
from sklearn.preprocessing import LabelEncoder, scale, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report                        , f1_score, accuracy_score, log_loss
from sklearn.feature_selection import SelectKBest                        , SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.decomposition import PCA                                , TruncatedSVD  #for sparse matrices
from sklearn.cluster import KMeans
from sklearn.grid_search import GridSearchCV 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Neural Nets
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU

# matplotlib 
import matplotlib.pyplot as plt

# GIS functionality
# import cartopy.io.shapereader as shpreader
import fiona
import pysal 
from pyproj import Proj
from pysal.cg.shapes import Point
from pysal.cg.locators import PolygonLocator, PointLocator, BruteForcePointLocator 
from pysal.cg.sphere import arcdist

# Multiprocessing 
import multiprocessing

# Logging
import logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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
        '''
        if self.columns is not None:
            for col in self.columns:
                if col.dtype not in numerics+ints:
                    output[col] = LabelEncoder().fit_transform(output[col])
                elif col.dtype not in ints:
                    output[col] = scale(output[col])
        else:
        '''
        try:
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
        except:
            output = LabelEncoder().fit_transform(output)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


logger.info('Load of data started')
train_raw = pd.read_csv('Data/train.csv')
#train_raw = pd.read_csv('Data/SFPD_Incidents_-_from_1_January_2003.csv')
#train_raw['Dates'] = pd.to_datetime(train_raw['Date']                                 +' '                                 +train_raw['Time']                                 ,format='%m/%d/%Y %H:%M')
test_raw = pd.read_csv('Data/test.csv')
logger.info('Load of data finished')
print train_raw.shape


# #### Deep dive into crimes dataset

# In[24]:

print train_raw.groupby(['Category']).size()


# In[25]:

def show_descripts(cat,first_n=10):
    ''' A function to evaluate descriptions for a category 
        sorted by the number of crimes
    '''
    g = train_raw[train_raw['Category']==cat]        [['Category','Descript']]        .groupby(['Category','Descript']).agg(len)
    for x in sorted(zip(g.index,g.values),key=lambda x: x[1], reverse=True)[:first_n]:
        print x

show_descripts('SUSPICIOUS OCC')


# I've noticed that categories aren't the best at describing the data in terms a model would understand. Some categories have a lot of crimes included that would make it difficult to have an accurate model, no matter what supplemental data sources are attached to the training set. Below I am looking to create a smaller number of 'meta-categories' that will be easier to predict by grouping individual categories and by including individual descriptions into appropriate meta-categories that will be easier for a model to interpret. 
# 
# After re-classifying the crimes for the training set, I will attempt to build a prediction model that will be used to predict meta-categories for the test set - which will then be utilized as an additional feature in the final model.

# In[26]:

def collar_crimes(x,y):
    ''' Add a meta category for 
        crimes based on the skills
        required
    '''
    blue_collar_violent = [ 
                   "ASSAULT"
                   , "KIDNAPPING"
                   , "ARSON"
                   , 'DOMESTIC VIOLENCE'
                   , 'GANG ACTIVITY'
                  ]
    blue_collar_other = [
                "VANDALISM"
                ,"DISORDERLY CONDUCT"
                ,"TRESPASS"
                ,'TREA'
               , 'LOITERING'
                ,'RESISTING ARREST'
                ,'PROBATION VIOLATION'
                ,'PROBATION VIOLATION'
                ,'VIOLATION OF RESTRAINING ORDER'
                ,'PAROLE VIOLATION'
    ]
    sex_crimes = [
            'SEX OFFENSES FORCIBLE',
            'PORNOGRAPHY/OBSCENE MAT',
            'SEX OFFENSES NON FORCIBLE',
            'PROSTITUTION'
        ]
    alcohol = [
        'DRIVING UNDER THE INFLUENCE',
        'DRUNKENNESS',
        'LIQUOR LAWS'
    ]
    drug = ['DRUG/NARCOTIC']
    theft = [
        'LARCENY/THEFT',
         'STOLEN PROPERTY',
         "ROBBERY",
         'CREDIT CARD, THEFT BY USE OF',
        'FRAUDULENT USE OF AUTOMATED TELLER CARD',
        'BURGLARY'
    ]
    vehicle = [
        'RECOVERED VEHICLE',
        'VEHICLE THEFT',
        'DRIVERS LICENSE, SUSPENDED OR REVOKED',
        'TRAFFIC VIOLATION',
        'TRAFFIC VIOLATION ARREST',
        'DRIVERS LICENSE, SUSPENDED OR REVOKED',
        'LOST/STOLEN LICENSE PLATE',
        'IMPOUNDED VEHICLE',
        'TRAFFIC ACCIDENT',
        'MALICIOUS MISCHIEF, VANDALISM OF VEHICLES'
    ]
    noncrime = [
        'MISSING PERSON',
        'RUNAWAY',
        'SUICIDE',
        'NON-CRIMINAL',
        'SUSPICIOUS OCC'
    ]
    white_collar = [ 
        "FRAUD"
       , "FORGERY/COUNTERFEITING"
       , "BAD CHECKS" 
       , "EXTORTION"
       , "EMBEZZLEMENT"
       , "BRIBERY"
        , 'CONSPIRACY'
    ]
    if x in blue_collar_violent or y in blue_collar_violent: return 1
    elif x in sex_crimes or y in sex_crimes: return 2
    elif x in alcohol or y in alcohol: return 3
    elif x in drug or y in drug: return 4
    elif x in theft or y in theft: return 5
    elif x in vehicle or y in vehicle: return 6
    elif x in noncrime or y in noncrime: return 7
    elif x in white_collar or y in white_collar: return 8
    elif x in blue_collar_other or y in blue_collar_other: return 9
    else: return 10
collar_crimes = np.vectorize(collar_crimes,otypes=[np.int64])
logger.info('Creation of collar_id started')
train_raw['collar_id'] = collar_crimes(train_raw['Category'],train_raw['Descript'])
logger.info('Creation of collar_id ended')


# In[27]:

def show_newcategories(col,first_n=10):
    ''' Evaluate how crimes are fit into the new 
        categories defined above
    '''
    g = train_raw[train_raw['collar_id']==col]        [['Category','Descript']]        .groupby(['Category','Descript']).agg(len)
    for x in sorted(zip(g.index,g.values),key=lambda x: x[1], reverse=True)[:first_n]:
        print x

show_newcategories(1)


# Another thing I've noticed is how some categories have minimal amounts of crimes which makes it difficult to build a model because there is such a strong bias in predicting crimes that occur the most often. I am creating a sampling methodology that samples with replacement in order to gather a more even amount of observations in each dataset.  

# In[28]:

# Gather counts of each category 
g = train_raw[['Category','Descript']].groupby(['Category']).agg(len)
group_cnts = pd.DataFrame({'Category':np.array(g.index).T,'Value':np.array(g.T)[0]})
print group_cnts


# ##### Add time of day 
# 
# Since a timestamp is not good for the generalization of the model, attempt bucketing of hours within each day, and break off day of month and year of crime and separate dimensions. 

# In[29]:

ceil = np.vectorize(math.ceil)
    
logger.info('Datetime conversion started')
train_raw['Dates'] = pd.to_datetime(train_raw['Dates'])
test_raw['Dates'] = pd.to_datetime(test_raw['Dates'])

logger.info('DaySegment creation started')
train_raw['DaySegment'] = ceil((train_raw['Dates'].dt.hour+1)/4).astype(np.int)
test_raw['DaySegment'] = ceil((test_raw['Dates'].dt.hour+1)/4).astype(np.int)

logger.info('TimeOfDay creation started')
train_raw['TimeOfDay'] = train_raw['Dates'].dt.hour
test_raw['TimeOfDay'] = test_raw['Dates'].dt.hour

logger.info('DayOfMonth creation started')
train_raw['DayOfMonth'] = train_raw['Dates'].dt.day
test_raw['DayOfMonth'] = test_raw['Dates'].dt.day

logger.info('Year creation started')
train_raw['Year'] = train_raw['Dates'].dt.year
test_raw['Year'] = test_raw['Dates'].dt.year

logger.info('Month creation started')
train_raw['Month'] = train_raw['Dates'].dt.month
test_raw['Month'] = test_raw['Dates'].dt.month

logger.info('YearQtr creation started')
train_raw['YearQtr'] = train_raw['Dates'].dt.year*100    +ceil(train_raw['Dates'].dt.month/4).astype(np.int)
test_raw['YearQtr'] = test_raw['Dates'].dt.year*100    +ceil(test_raw['Dates'].dt.month/4).astype(np.int)

logger.info('YearSegment creation started')
train_raw['YearSegment'] = ceil(train_raw['Dates'].dt.month/4).astype(np.int)
test_raw['YearSegment'] =  ceil(test_raw['Dates'].dt.month/4).astype(np.int)
logger.info('Date feature processing ended')


# ##### Add clustering based on lat/lon and time of day

# Currently, by using the exact location of the crime, the model does not generalize very well. I KMeans to segment training data into clusters based on location, time of day, and year and add both the cluster label and distance from centroid as features.

# In[30]:

def test_clusters(range_n_clusters,fields=['X','Y','YearSegment','Year']):
    km_models = []
    i_scores = []
    le = MultiColumnLabelEncoder()
    nrm = StandardScaler()
    tr = train_raw[['X','Y','TimeOfDay','YearQtr']].copy()
    pl = Pipeline([('le',le),('nrm',nrm)])
    tr = pl.fit_transform(tr)
    logger.info('Cluster test starting')
    for n_clusters in range(range_n_clusters):
        if n_clusters>1:
            km = KMeans(n_clusters=n_clusters, random_state=5)
            km.fit(tr)
            km_models.append(km)
            inertia = km.inertia_ 
            print 'For {0}, inertia = {1}'.format(
                n_clusters, inertia
            )
            i_scores.append(inertia)
       
    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('KMeans inertia values')
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Inertia')
    ax.plot([i for i in range(range_n_clusters)              if i>1],i_scores,'-', linewidth=2)
    plt.show()
    
# test_clusters(25)


# Because of the clustering process, I found that there are a few points with what must be default values of lat/lon coordinates: 90,-120.5. Those are values that aren't interpretable by GIS packages and cause significant issues with clustering as well, so I have to manually impute them with better default values here.

# In[31]:

# Manually impute bad X,Y values as 
train_raw.loc[train_raw['X']==-120.5,['X']] = np.mean(train_raw['X'])
train_raw.loc[train_raw['Y']==90,['Y']] = np.mean(train_raw['Y'])
test_raw.loc[test_raw['X']==-120.5,['X']] = np.mean(test_raw['X'])
test_raw.loc[test_raw['Y']==90,['Y']] = np.mean(test_raw['Y'])

# In[32]:

# Reload data in case of changes
logger.info('Clustering started')
le = MultiColumnLabelEncoder()
nrm = StandardScaler()
tr = train_raw[['X','Y','TimeOfDay','YearQtr']].copy()
pl = Pipeline([('le',le),('nrm',nrm)])
tr = pl.fit_transform(tr)

# Set k
k = 20

# Initialize Kmeans model
km = KMeans(n_clusters=k)
logger.info('Clustering training data')
train_raw['KMcluster'] = km.fit_predict(tr)

# Calculate distances
logger.info('Clustering distance calculation for training data')
distances = km.transform(tr)
train_raw['KMdistance'] = np.min(distances,axis=1)

# Predict for test dataset
logger.info('Clustering training data')
tr = test_raw[['X','Y','TimeOfDay','YearQtr']].copy()
tr = pl.transform(tr)
test_raw['KMcluster'] = km.predict(tr)
logger.info('Clustering distance calculation for training data')
distances = km.transform(tr)
test_raw['KMdistance'] = np.min(distances,axis=1)
logger.info('Clustering finished')


# In[33]:

def show_clustercat(cluster):
    ''' A function that shows top crimes in each cluster '''
    g = train_raw[train_raw['KMcluster']==cluster].        groupby(['Category','KMcluster'])['Category'].agg(len)
    for x in sorted(zip(g.index,g.values),key=lambda x:                     (x[0][1],x[1]), reverse=True)[:10]:
        print x
    
show_clustercat(11)


# In[34]:

def plot_clusters():
    ''' Show all clusters in individual scatterplots '''
    f, axarr = plt.subplots(4, 5, sharex=True, sharey=True)
    le = MultiColumnLabelEncoder()
    nrm = StandardScaler()
    tr = train_raw[['X','Y','TimeOfDay','YearQtr']].copy()
    pl = Pipeline([('le',le),('nrm',nrm)])
    tr = pl.fit_transform(tr)
    pca = PCA(n_components=2)
    X = pca.fit_transform(tr)
    K = np.array(train_raw['KMcluster'])
    for i in range(20):
        if i<5: e=0
        elif i<10: e=1
        elif i<15: e=2
        else: e=3
        z = i - 5*e
        axarr[e, z].plot(X[K==i,0]                            ,X[K==i,1]                            ,'bo')
        axarr[e, z].set_title('K:{}'.format(i))
        plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plot_clusters()


# Add address
#

def transform_address(x):
    x = re.sub(r'[0-9\/]','',x)
    x = x.upper()
    x = x.split()
    x = sorted(set(x))
    #removes = ['OF',' ','AV','ST','BL','CT','WY'\
    #           ,'DR','PL','RD','LN','TR','CR','TH'\
    #           ,'THE','BLOCK','HY','BLVD']
    removes = ['OF','THE','TH',' ']
    x = [ z for z in x if z not in removes ]
    x = ' '.join(x)
    x = re.sub(r'\ \ ',' ',x)
    x = re.sub(r'^\ ','',x)
    return x

transform_address = np.vectorize(transform_address)
train_raw['AddressMod'] = transform_address(train_raw['Address'])
test_raw['AddressMod'] = transform_address(test_raw['Address'])



# #### Test model
# 

# In[35]:

model_fields = [#'Category',
                 'DayOfWeek',
                 'PdDistrict',
                 'DaySegment',
                 'TimeOfDay',
                 'DayOfMonth',
                 'Year',
                 'YearSegment',
                 'X',
                 'Y',
                 'KMcluster',
                 'KMdistance']
categorical_fields = [#'Category',
                 'DayOfWeek',
                 'PdDistrict',
                 'DaySegment',
                 'TimeOfDay',
                 'DayOfMonth',
                 'Year',
                 'YearSegment',
                 'KMcluster']


# In[36]:

logger.info('Sampling started')
tr = train_raw[model_fields+['Category']].copy().iloc[                   np.random.permutation(len(train_raw))]

logger.info('Creating dev and test datasets')
dev_train    , dev_train_labels_cat = tr[model_fields][50001:],                                tr['Category'][50001:]
dev_test    , dev_test_labels_cat = tr[model_fields][:50000],                            tr['Category'][:50000]


# In[37]:

def make_pipeline(model_fields,categorical_fields):
    ''' Create pipeline that will be 
        used multiple times.
        
        Run test model to see what performance is 
        with current set of features. 
    '''
    le = MultiColumnLabelEncoder()
    cf = [i for i,x in enumerate(model_fields) if x in categorical_fields]
    ohe = OneHotEncoder(categorical_features=cf,sparse=True)
    svd = TruncatedSVD(n_components=10) 
    ss = StandardScaler()
    lr = LogisticRegression(C=0.01,solver='lbfgs'                            ,multi_class='multinomial')
    pl = Pipeline([('le',le)        # Recode text features as integers with LabelEncoder
                   ,('ohe',ohe)     # Create dummy features for each categorical feature
                   ,('svd',svd)     # Decompose features into a smaller projection \
                                    #   as a dense matrix for scaler 
                   ,('ss',ss)       # Scale numerical variables 
                   ,('lr',lr)])     # Predict with LogisticRegression
                   
    return pl

pl = make_pipeline(model_fields,categorical_fields)


# In[38]:

def predict_category(pl):
    '''Predict the category of crime
    '''
    
    logger.info('Fitting training data') 
    pl.fit(dev_train, dev_train_labels_cat)
    return pl

logger.info('1st model for category started')
# pl_category = predict_category(pl)
logger.info('1st model for category ended')


# In[39]:


# ### Part 3: Get data from other sources
# 
# SF OpenData has a ton of supplemental data sources that will be great to try out for this effort.  
# 
# NOTE: one of them is actually a list of crimes that seems to match data in the training set. I will NOT use that data to train my model or match against the test dataset; however, I believe that many contestants are doing this, given that there is a very clear separation in scores that indicates to me that maybe there is some cheating happening.

# In[40]:

# http://spatialreference.org/ref/epsg/2227/
p = Proj('+proj=lcc +init=EPSG:2227 +datum=NAD83 +units=us-ft +no_defs',preserve_units=True)
convert_vals = np.vectorize(lambda x,y: p(x,y))
convert_vals_inv = np.vectorize(lambda x,y: p(x,y,inverse=True))


# In[41]:

shpfilename_elect = 'Data/SanFranciscoElectricityUse/SanFranciscoElectricityUse.shp'
shpfilename_school = 'Data/schools_public_pt/schools_public_pt.shp'
shpfilename_neighborhoods = 'Data/planning_neighborhoods/planning_neighborhoods.shp'
shpfilename_jobdensity = 'Data/SanFranciscoJobDensity/SanFranciscoJobDensity.shp'
shpfilename_income = 'Data/SanFranciscoIncome/SanFranciscoIncome.shp'
shpfilename_sfpdplots = 'Data/sfpd_plots/sfpd_plots.shp'
shpfilename_sfpdsectors = 'Data/sfpd_sectors/sfpd_sectors.shp'
shpfilename_employment = 'Data/SanFranciscoEmploymentRate/SanFranciscoEmploymentRate.shp'
shpfilename_speeding = 'Data/SanFranciscoSpeedLimitCompliance/SanFranciscoSpeedLimitCompliance.shp'
street_tree_locations = 'Data/Street_Tree_List.csv'
business_locations = 'Data/Registered_Business_Map.csv'
park_locations = 'Data/Park_and_Open_Space_Map.csv'
offstreet_parking_locations = 'Data/Off-street_parking_lots_and_parking_garages_map.csv'


# #### Convert lat/lon to coordinates that match shape files

# In[42]:

train_raw['New_X'], train_raw['New_Y'] =     convert_vals(train_raw['X'],train_raw['Y'])
test_raw['New_X'], test_raw['New_Y'] =     convert_vals(test_raw['X'],test_raw['Y'])


# In[43]:

pair_fields = ['New_X','New_Y','X','Y']
base_pairs = pd.concat([train_raw[pair_fields]                                 ,test_raw[pair_fields]])
logger.info('Base pairs prior to duplicate removal: {}'.format(base_pairs.shape[0]))
base_pairs = base_pairs.drop_duplicates()
logger.info('Base pairs after duplicate removal: {}'.format(base_pairs.shape[0]))
base_points = [ Point((x,y)) for x,y in zip(base_pairs['New_X'],base_pairs['New_Y']) ]


# #### Read and process CSVs

# In[44]:

'''
trees = pd.read_csv(street_tree_locations)
businesses = pd.read_csv(business_locations)
parks = pd.read_csv(park_locations, quotechar = "\"")
offstreet_parking = pd.read_csv(offstreet_parking_locations)
'''


# In[45]:

def getll(x):
    ''' Get lat/lon for 311 and parking file '''
    try:
        x = re.sub(r'[\(\)]','',x)
        x = x.split(', ')
        return float(x[0]),float(x[1])
    except:
        return (None,None)


# In[46]:

'''
offstreet_parking['LatLon'] = offstreet_parking['Location 1'].apply(getll)
offstreet_parking['X'],offstreet_parking['Y'] = \
    offstreet_parking['LatLon'].apply(lambda x: x[1])\
    ,offstreet_parking['LatLon'].apply(lambda x: x[0])
offstreet_parking['XCoord'], offstreet_parking['YCoord'] = \
    convert_vals(offstreet_parking['X'],offstreet_parking['Y'])
'''


# In[47]:

def getll(x):
    ''' Get lat/lon for park & business file '''
    try:
        x = x.split('\n')[2]
        x = re.sub(r'[\(\)]','',x)
        x = x.split(', ')
        return float(x[0]),float(x[1])
    except:
        return (None,None)


# In[48]:

'''
parks['LatLon'] = parks['Location 1'].apply(getll)
parks['X'] = parks['LatLon'].apply(lambda x: x[1])
parks['Y'] = parks['LatLon'].apply(lambda x: x[0])
parks['XCoord'], parks['YCoord'] = convert_vals(parks['X'],parks['Y'])
'''


# In[49]:

'''
businesses['LatLon'] = businesses['Business_Location'].apply(getll)
businesses['X'] = businesses['LatLon'].apply(lambda x: x[1])
businesses['Y'] = businesses['LatLon'].apply(lambda x: x[0])
businesses['XCoord'], businesses['YCoord'] = convert_vals(businesses['X'],businesses['Y'])
businesses_class02 = businesses.loc[businesses['Class Code']=='02',['XCoord','YCoord']]
businesses_class07 = businesses.loc[businesses['Class Code']=='07',['XCoord','YCoord']]
businesses_class08 = businesses.loc[businesses['Class Code']=='08',['XCoord','YCoord']]
'''


# ### Functions to find matches to crimes from supplemental data sources, based on location
# 
# NOTE: Ideally these data sources would have some temporal element as well; however, most of them only contain data for a single time snapshot that is roughly in the middle of all the crimes data - better than nothing, but not ideal.

# #### Read shapefiles and store properties into pandas dataframe

# In[50]:

def read_speeding():
    ''' Process speeding rates into pandas dataframe '''
    shp = fiona.open(shpfilename_speeding)
    n = len(shp)
    Over_pct,    O5mph_pct,    Speed_avg =         np.empty(n,dtype=np.float),        np.empty(n,dtype=np.float),        np.empty(n,dtype=np.float)
    for i,s in enumerate(shp):
        Over_pct[i] = s['properties']['Over_pct']
        O5mph_pct[i] = s['properties']['O5mph_pct']
        Speed_avg[i] = s['properties']['Speed_avg']
    shp.close()
    props_df = pd.DataFrame({
            'Id':[i+1 for i in range(n)],\
            'Over_pct':Over_pct,\
            'O5mph_pct':O5mph_pct,\
            'Speed_avg':Speed_avg
        })
    return props_df

# props_df_speeding = read_speeding()


# In[51]:

def read_employment():
    ''' Process employment rates into pandas dataframe '''
    shp = fiona.open(shpfilename_employment)
    n = len(shp)
    Employ_pct,    Employ_moe =         np.empty(n,dtype='|S25'),        np.empty(n,dtype='|S25')
    for i,s in enumerate(shp):
        Employ_pct[i] = s['properties']['Employ_pct']
        Employ_moe[i] = s['properties']['Employ_moe']
    shp.close()
    props_df = pd.DataFrame({
            'Id':[i+1 for i in range(n)],\
            'Employ_pct':Employ_pct,\
            'Employ_moe':Employ_moe
        })
    return props_df

props_df_employment = read_employment()


# In[52]:

def read_sfpdsectors():
    ''' Process SFPD sectors into pandas dataframe '''
    shp = fiona.open(shpfilename_sfpdsectors)
    n = len(shp)
    SECTORID =         np.empty(n,dtype='|S25')
    for i,s in enumerate(shp):
        SECTORID[i] = s['properties']['SECTORID']
    shp.close()
    props_df = pd.DataFrame({
            'Id':[i+1 for i in range(n)],
            'SECTORID':SECTORID
        })
    return props_df

props_df_sfpdsectors = read_sfpdsectors()


# In[53]:

def read_sfpdplots():
    ''' Process SFPD plot into pandas dataframe '''
    shp = fiona.open(shpfilename_sfpdplots)
    n = len(shp)
    PLOT =         np.empty(n,dtype='|S25')
    for i,s in enumerate(shp):
        PLOT[i] = s['properties']['PLOT']
    shp.close()
    props_df = pd.DataFrame({
            'Id':[i+1 for i in range(n)],
            'PLOT':PLOT
        })
    return props_df

props_df_sfpdplots = read_sfpdplots()


# In[54]:

def read_neighborhoods():
    ''' Process neighborhood file into pandas dataframe '''
    shp = fiona.open(shpfilename_neighborhoods)
    n = len(shp)
    NEIGHBORHO =         np.empty(n,dtype='|S25')
    for i,s in enumerate(shp):
        NEIGHBORHO[i] = s['properties']['neighborho']
    shp.close()
    props_df = pd.DataFrame({
            'Id':[i+1 for i in range(n)],
            'NEIGHBORHO':NEIGHBORHO
        })
    return props_df

props_df_neighborhood = read_neighborhoods()


# In[55]:

def read_income():
    ''' Process income file into pandas dataframe '''
    shp = fiona.open(shpfilename_income)
    n = len(shp)
    MedInc_d,    MedInc_moe,    pC_Inc_d,    pC_Inc_moe =         np.empty(n,dtype='|S25'),        np.empty(n,dtype='|S25'),        np.empty(n,dtype='|S25'),        np.empty(n,dtype='|S25')
    for i,s in enumerate(shp):
        MedInc_d [i] = s['properties']['MedInc_d']
        MedInc_moe [i] = s['properties']['MedInc_moe']
        pC_Inc_d [i] = s['properties']['pC_Inc_d']
        pC_Inc_moe [i] = s['properties']['pC_Inc_moe']
    shp.close()
    props_df = pd.DataFrame({
            'Id':[i+1 for i in range(n)],
            'MedInc_d':MedInc_d,
            'MedInc_moe':MedInc_moe,
            'pC_Inc_d':pC_Inc_d,
            'pC_Inc_moe':pC_Inc_moe
        })
    return props_df

props_df_income = read_income()


# In[56]:

def read_jobdensity():
    ''' Process job density file into pandas dataframe '''
    shp = fiona.open(shpfilename_jobdensity)
    n = len(shp)
    JOBS_PSMI,    JOBS_CNT =         np.empty(n,dtype='|S25'),        np.empty(n,dtype='|S25')
    for i,s in enumerate(shp):
        JOBS_PSMI [i] = s['properties']['Jobs_psmi']
        JOBS_CNT [i] = s['properties']['Jobs_cnt']
    shp.close()
    props_df = pd.DataFrame({
            'Id':[i+1 for i in range(n)],
            'JOBS_PSMI':JOBS_PSMI,
            'JOBS_CNT':JOBS_CNT
        })
    return props_df

props_df_jobs = read_jobdensity()


# In[57]:

def read_schoolfile():
    ''' Process school file into pandas dataframe '''
    shp = fiona.open(shpfilename_school)
    n = len(shp)
    SCHOOL_TYP,    DEPT,    FACILITY_N,    DEPTNAME,    FACILITY_I =         np.empty(n,dtype='|S25'),        np.empty(n,dtype='|S25'),        np.empty(n,dtype='|S25'),        np.empty(n,dtype='|S25'),        np.empty(n,dtype='|S25')
    for i,s in enumerate(shp):
        SCHOOL_TYP[i] = s['properties']['SCHOOL_TYP']
        DEPT [i] = s['properties']['DEPT']
        FACILITY_N [i] = s['properties']['FACILITY_N']
        DEPTNAME [i] = s['properties']['DEPTNAME']
        FACILITY_I [i] = s['properties']['FACILITY_I']
    shp.close()
    props_df = pd.DataFrame({
            'Id':[i+1 for i in range(n)],
            'SCHOOL_TYP':SCHOOL_TYP
        })
    return props_df

props_df_schools = read_schoolfile()


# In[58]:

def read_electfile():
    shp = fiona.open(shpfilename_elect)
    n = len(shp)
    kWh_pC,    kWh,    Zip,    Pop2010_zc =         np.empty(n,dtype=np.float64),        np.empty(n,dtype=np.float64),        np.empty(n,dtype='|S10'),        np.empty(n,dtype=np.int64)
    for i,s in enumerate(shp):
        kWh_pC[i] = s['properties']['kWh_pC']
        kWh [i] = s['properties']['kWh']
        Zip [i] = s['properties']['Zip']
        Pop2010_zc [i] = s['properties']['Pop2010_zc']
    shp.close()

    props_df_elect = pd.DataFrame({
            'Id':[i+1 for i in range(n)],\
            'kWh_pC':kWh_pC,\
            'kWh':kWh,\
            'Zip':Zip,\
            'Pop2010_zc':Pop2010_zc\
        })
    return props_df_elect

props_df_elect = read_electfile()


# ##### Polygon search functions

# In[59]:

def polygon_search(shpfilename):
    ''' Iterate through shapefile polygons
        and find id of polygon for 
        each datapoint if it fits inside 
        of the polygon boundaries
    '''
    shp = pysal.open(shpfilename,'r')
    pl = PolygonLocator([p for p in shp])
    shp.close()
    return pl


# ##### Centroid search function (faster than polygon search)

# In[60]:

def coord_search_centroid(shpfile,                 locator_fun=BruteForcePointLocator):
    ''' Since polygon search is not very efficient
        when there are many polygons, instead
        do a comparison to each polygon centroid 
    '''
    logger.info('Gathering centroids')
    gather_centroids = lambda shp: [p.centroid for p in shp]

    # Read file
    shp = pysal.open(shpfile,'r')
    centroids = gather_centroids(shp)
    pl = locator_fun(centroids)
    shp.close()
    
    return pl,centroids


# ##### Functions that iterate over each observation and match GIS data points
# These functions are used to divide the workload among multiple processors. Since the tasks are very CPU-bound and can be run separately then compiled afterward, this works quite well.

# In[61]:

def run_iters_points(points, point_locator,               proximity=100,log_at=100000):
    ''' Iterate through points and return number
        of points in surrounding proximity
    '''
    point_fun = lambda x,pl: pl.proximity(x,proximity)
    surrounding_pts = np.zeros(len(points),dtype=np.int64)
    for i,p in enumerate(points):
        if i%log_at==0: logger.info('running {0} row'.format(i))
        pts = point_fun(p,point_locator)
        surrounding_pts[i] = len(pts)
    return surrounding_pts


# In[62]:

def run_iters_point_distance(points, point_locator            ,log_at=100000):
    ''' Iterate through points and return distance
        to the nearest point
    '''
    point_fun = lambda x,pl: pl.nearest(x)
    distances = np.zeros(len(points),dtype=np.int64)
    for i,p in enumerate(points):
        if i%log_at==0: logger.info('running {0} row'.format(i))
        pt = point_fun(p,point_locator)
        distances[i] = arcdist(p,pt)
    return distances


# In[63]:

def run_iters_poly(points, point_locator                   ,log_at=100000):
    ''' Iterate through points and find matching polygon '''
    
    def return_poly_id(pl,x):
        ''' Find the polygon within the 
            PolygonLocator that 
            matches to each point
        '''
        try:
            return pl.contains_point(x)[0].id
        except IndexError:
            return -1
    
    poly_ids = np.zeros(len(points),dtype=np.int64)
    for i,p in enumerate(points):
        if i%log_at==0: logger.info('running {0} row'.format(i))
        poly_ids[i] = return_poly_id(point_locator,p)
    return poly_ids


# In[64]:

def run_iters_centroid(points, point_locator, centroids,               log_at=1000):
    ''' Iterate through points to find 
        the nearest matching polygon centroid point
        
        Faster than running polygon search 
    '''
    logger.info('Running iterations on {} points'.format(len(points)))
    point_fun = lambda x,pl: pl.nearest(x)
    id_fun = lambda p, centroids: [i for i,c in enumerate(centroids)                                  if c==p][0]
    nearest_ids = np.zeros(len(points),dtype=np.int64)
    for i,p in enumerate(points):
        if i%log_at==0: logger.info('running {0} row'.format(i))
        pt = point_fun(p,point_locator)
        pt_id = id_fun(pt, centroids)
        nearest_ids[i] = pt_id
    return nearest_ids


# ##### Multiprocessing function to distribute work over many cores

# In[65]:

def do_workload(worker,base_points,n_threads=2):
    ''' Create subprocess threads and combine work 
        after finishing.
        
        NOTE: freezes in ipython notebook
    '''
    n = n_threads
    # multiprocessing.freeze_support()
    pool = multiprocessing.Pool(n)
    
    p = [ i*len(base_points)//n for i in range(n+1) ]

    points_list = [ base_points[p[i]:p[i+1]] for i in range(n) ]
    points_list += [base_points[p[n]:]]

    res = pool.map(worker, points_list)
    pool.close()
    pool.join()
    x = pd.concat(res,axis=0)
    return x


# #### Match crimes data to supplementary data sources
# 
# The core of the work below was done on other machines, or on my laptop but outside of ipython notebook due to issues with parallel processing. In most cases, the code that was run is commented out, other than reading in the data from the files that were already processed. 
# 
# Most the matching is done based on lat/lon coordinates, which are not at all unique for each crime. In fact, only about 1.5% of the lat/lon coordinates are unique compared to the total number of crimes in the file.

# ##### Find nearby schools

# In[66]:

def search_schools(pts):
    ''' Find nearest point and measure distance
        for every datapoint 
    '''
    hs = props_df_schools[props_df_schools['SCHOOL_TYP']=='High School']['Id']
    cs = props_df_schools[props_df_schools['SCHOOL_TYP']=='County School']['Id']
    chs = props_df_schools[props_df_schools['SCHOOL_TYP']=='Charter School']['Id']
    ms = props_df_schools[props_df_schools['SCHOOL_TYP']=='Middle School']['Id']
    em = props_df_schools[props_df_schools['SCHOOL_TYP']=='Elementary']['Id']
    
    shp = pysal.open(shpfilename_school,'r')
    pl_hs = BruteForcePointLocator([p for p in shp if p.id in hs])
    pl_cs = BruteForcePointLocator([p for p in shp if p.id in cs])
    pl_chs = BruteForcePointLocator([p for p in shp if p.id in chs])
    pl_ms = BruteForcePointLocator([p for p in shp if p.id in ms])
    pl_em = BruteForcePointLocator([p for p in shp if p.id in em])
    shp.close()

    return_point_hs = lambda x: pl_hs.nearest(x)
    return_point_cs = lambda x: pl_cs.nearest(x)
    return_point_chs = lambda x: pl_chs.nearest(x)
    return_point_ms = lambda x: pl_ms.nearest(x)
    return_point_em = lambda x: pl_em.nearest(x)
    
    # point_ids = np.zeros(len(points),dtype=np.int8)
    point_distances_hs = np.zeros(len(pts),dtype=np.float64)
    point_distances_cs = np.zeros(len(pts),dtype=np.float64)
    point_distances_chs = np.zeros(len(pts),dtype=np.float64)
    point_distances_ms = np.zeros(len(pts),dtype=np.float64)
    point_distances_em = np.zeros(len(pts),dtype=np.float64)
    
    def run_iters():
        for i,p in enumerate(pts):
            if i%1000==0: logger.info('running {0} row'.format(i))
            pt_hs = return_point_hs(p)
            pt_cs = return_point_cs(p)
            pt_chs = return_point_chs(p)
            pt_ms = return_point_ms(p)
            pt_em = return_point_em(p)
            
            point_distances_hs[i] = arcdist(p,pt_hs)
            point_distances_cs[i] = arcdist(p,pt_cs)
            point_distances_chs[i] = arcdist(p,pt_chs)
            point_distances_ms[i] = arcdist(p,pt_ms)
            point_distances_em[i] = arcdist(p,pt_em)

    run_iters()
    
    return point_distances_hs,            point_distances_cs,            point_distances_chs,            point_distances_ms,            point_distances_em

'''
point_distances_hs,\
            point_distances_cs,\
            point_distances_chs,\
            point_distances_ms,\
            point_distances_em = search_schools(base_points)
schools = pd.DataFrame({'high school nearby':point_distances_hs\
                       ,'county school nearby':point_distances_cs\
                       ,'charter school nearby':point_distances_chs\
                       ,'middle school nearby':point_distances_ms\
                       ,'elementary school nearby':point_distances_em})
schools['X'],schools['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
schools.to_csv('Data/schools.csv')
'''
schools = pd.read_csv('Data/schools.csv',header=0,index_col=0)


# ##### Find nearby parking lots

# In[67]:

filename='Data/parking.csv'
'''
offstreet_parking_points = [ Point((x,y)) for x,y in zip(offstreet_parking['XCoord'],offstreet_parking['YCoord']) ]
def worker(points):
    return pd.DataFrame({'nearest_parkinglot_distance':run_iters_point_distance(points\
                                  ,BruteForcePointLocator(offstreet_parking_points)\
                                  ,log_at=100000)})
logger.info('Starting work on parking lots')
x = do_workload(worker,base_points,8)
x['X'],x['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x.to_csv(filename,index=True)
'''
parkinglots = pd.read_csv(filename,index_col=0,header=0)


# ##### Process park CSV and find if park is nearby 

# In[68]:

filename='Data/parks.csv'
'''
park_points = [ Point((x,y)) for x,y in zip(parks['XCoord'],parks['YCoord']) ]
def worker(points):
    return pd.DataFrame({'nearest_park_distance':run_iters_point_distance(points\
                                  ,BruteForcePointLocator(park_points)\
                                  ,log_at=100000)})
logger.info('Starting work on parks')
x = do_workload(worker,base_points,8)
x['X'],x['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x.to_csv(filename,index=True)
'''
parks = pd.read_csv(filename,index_col=0,header=0)


# ##### Process major business classes nearby 

# In[69]:

filename02='Data/business_classes02.csv'
filename07='Data/business_classes07.csv'
filename08='Data/business_classes08.csv'
'''
bus02_points = [ Point((x,y)) for x,y in zip(businesses_class02['XCoord'],businesses_class02['YCoord']) ]
bus07_points = [ Point((x,y)) for x,y in zip(businesses_class07['XCoord'],businesses_class07['YCoord']) ]
bus08_points = [ Point((x,y)) for x,y in zip(businesses_class08['XCoord'],businesses_class08['YCoord']) ]
def worker02(points):
    return pd.DataFrame({'nearest_business_distance_02class':run_iters_point_distance(points\
                                  ,BruteForcePointLocator(bus02_points)\
                                  ,log_at=1000)})
def worker07(points):
    return pd.DataFrame({'nearest_business_distance_07class':run_iters_point_distance(points\
                                  ,BruteForcePointLocator(bus07_points)\
                                  ,log_at=1000)})
def worker08(points):
    return pd.DataFrame({'nearest_business_distance_08class':run_iters_point_distance(points\
                                  ,BruteForcePointLocator(bus08_points)\
                                  ,log_at=1000)})

x02 = do_workload(worker02,base_points,8)
x02['X'],x02['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x02.to_csv(filename02,index=True)

x07 = do_workload(worker07,base_points,8)
x07['X'],x07['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x07.to_csv(filename07,index=True)

x08 = do_workload(worker08,base_points,8)
x08['X'],x08['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x08.to_csv(filename08,index=True)
'''
business_class02 = pd.read_csv(filename02,index_col=0,header=0)
business_class07 = pd.read_csv(filename07,index_col=0,header=0)
business_class08 = pd.read_csv(filename08,index_col=0,header=0)


# ##### Find neighborhood of crime

# In[70]:

filename='Data/neighborhoods.csv'
'''
neighborhood_locator = polygon_search(shpfilename_neighborhoods)
def worker(points):
    return pd.DataFrame({'neighborhood_id':run_iters_poly(points\
                                    ,neighborhood_locator
                                    ,log_at=10000)})

logger.info('Starting work on neighborhood')
x = do_workload(worker,base_points,8)
x['X'],x['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x.to_csv(filename,index=True)
'''
neighborhoods = pd.read_csv(filename,index_col=0,header=0)


# ##### Add SFPD Sectors

# In[71]:

filename='Data/sfpd_sectors.csv'
'''
sfpdsector_locator = polygon_search(shpfilename_sfpdsectors)
def worker(points):
    return pd.DataFrame({'sfpd_sector_id':run_iters_poly(points\
                                    ,sfpdsector_locator
                                    ,log_at=10000)})

logger.info('Starting work on SFPD Sectors')
x = do_workload(worker,base_points,8)
x['X'],x['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x.to_csv(filename,index=True)
'''
sfpd_sectors = pd.read_csv(filename,index_col=0,header=0)


# ##### Add SFPD plots

# In[72]:

filename='Data/sfpd_plots.csv'
'''
sfpdplot_locator = polygon_search(shpfilename_sfpdplots)
def worker(points):
    return pd.DataFrame({'sfpd_plot_id':run_iters_poly(points\
                                    ,sfpdplot_locator
                                    ,log_at=10000)})

logger.info('Starting work on SFPD Plots')
x = do_workload(worker,base_points,8)
x['X'],x['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x.to_csv(filename,index=True)
'''
sfpd_plots = pd.read_csv(filename,index_col=0,header=0)


# ##### Process tree CSV and find number of trees nearby to crime

# In[73]:

filename='Data/trees_wxy.csv'
''' 
tree_points = [ Point((x,y)) for x,y in zip(trees['XCoord'],trees['YCoord']) ]

def worker(points):
    return pd.DataFrame({'trees':run_iters_points(points\
                                  ,BruteForcePointLocator(tree_points)\
                                  ,proximity=100
                                  ,log_at=1000)})

x = do_workload(worker,base_points,8)
x.to_csv(filename,index=True)
'''
trees = pd.read_csv(filename,index_col=0,header=0)


# ##### Process jobs data

# In[74]:

filename = 'Data/job_ids.csv'
'''
jobzone_locator = polygon_search(shpfilename_jobdensity)
def worker(points):
    return pd.DataFrame({'job_id':run_iters_poly(points\
                                    ,jobzone_locator
                                    ,log_at=10000)})

logger.info('Starting work on job zones')
x = do_workload(worker,base_points,8)
x['X'],x['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x.to_csv(filename,index=True)
'''
jobs = pd.read_csv(filename,index_col=0,header=0)


# ##### Process employment data

# In[75]:

filename = 'Data/employment.csv'
'''
employment_locator = polygon_search(shpfilename_employment)
def worker(points):
    return pd.DataFrame({'employment_id':run_iters_poly(points\
                                    ,employment_locator
                                    ,log_at=10000)})

logger.info('Starting work on employment zones')
x = do_workload(worker,base_points,8)
x['X'],x['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x.to_csv(filename,index=True)
'''
employment = pd.read_csv(filename,index_col=0,header=0)


# ##### Process income data

# In[76]:

filename='Data/income.csv'
'''
income_locator,centroids = coord_search_centroid(shpfilename_income)

def worker(points):
    return pd.DataFrame({'income_id':run_iters_centroid(points\
                                    ,income_locator
                                    ,centroids
                                    ,log_at=10000)})

logger.info('Starting work on incomes')
x = do_workload(worker,base_points,8)
x['X'],x['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x.to_csv(filename,index=True)
'''
incomes = pd.read_csv(filename,index_col=0,header=0)


# ##### Process electricity usage data

# In[77]:

filename='Data/electricity.csv'
'''
elect_locator,centroids = coord_search_centroid(shpfilename_elect)

def worker(points):
    return pd.DataFrame({'electricity_id':run_iters_centroid(points\
                                    ,elect_locator
                                    ,centroids
                                    ,log_at=10000)})

logger.info('Starting work on electricity')
x = do_workload(worker,base_points,8)
x['X'],x['Y'] = np.array(base_pairs['X']),np.array(base_pairs['Y'])
x.to_csv(filename,index=True)
'''
electricity = pd.read_csv(filename,index_col=0,header=0)


# #### Add new columns
# Datasets matched by location:
# * incomes
# * employment
# * jobs
# * trees
# * neighborhoods
# * business_class02
# * business_class07
# * business_class08
# * parks
# * schools
# * sfpd_plots
# * sfpd_sectors
# * electricity
# 

# In[78]:

keep_fields = [#'IncidntNum',
                 'Category',
                 #'Descript',
                 'DayOfWeek',
                 #'Date',
                 #'Time',
                 'PdDistrict',
                 #'Resolution',
                 #'Address',
                 'X',
                 'Y',
                 #'Location',
                 #'PdId',
                 'Dates',
                 'collar_id',
                 'DaySegment',
                 'TimeOfDay',
                 'DayOfMonth',
                 'Year',
                 'Month',
                 'Year',
                 'YearSegment',
                 'KMcluster',
                 'KMdistance',
                 #'New_X',
                 #'New_Y',
		 'AddressMod'
              ]
train_raw = train_raw[keep_fields]


# In[79]:


incomes = incomes.drop_duplicates(subset=['X','Y'],take_last=True)
employment = employment.drop_duplicates(subset=['X','Y'],take_last=True)
jobs = jobs.drop_duplicates(subset=['X','Y'],take_last=True)
trees = trees.drop_duplicates(subset=['X','Y'],take_last=True)
neighborhoods = neighborhoods.drop_duplicates(subset=['X','Y'],take_last=True)
business_class02 = business_class02.drop_duplicates(subset=['X','Y'],take_last=True)
business_class07 = business_class07.drop_duplicates(subset=['X','Y'],take_last=True)
business_class08 = business_class08.drop_duplicates(subset=['X','Y'],take_last=True)
parks = parks.drop_duplicates(subset=['X','Y'],take_last=True)
schools = schools.drop_duplicates(subset=['X','Y'],take_last=True)
sfpd_plots = sfpd_plots.drop_duplicates(subset=['X','Y'],take_last=True)
sfpd_sectors = sfpd_sectors.drop_duplicates(subset=['X','Y'],take_last=True)
electricity = electricity.drop_duplicates(subset=['X','Y'],take_last=True)


# In[80]:

def create_mergekey(df):
    ''' I had problems using pandas
        merge on two float keys,
        so I am creating a single 
        string mergekey and it works   
    ''' 
    return (df['X'] * 10000000000).astype(str)            ,(df['Y'] * 10000000000).astype(str)

def mergekeys():
    logger.info('Creating mergekeys')
    train_raw['X_merge'],train_raw['Y_merge'] = create_mergekey(train_raw)
    train_raw['Datemerge'] = train_raw['Dates'].astype(str)
    test_raw['X_merge'],test_raw['Y_merge'] = create_mergekey(test_raw)
    test_raw['Datemerge'] = test_raw['Dates'].astype(str)
    incomes['X_merge'],incomes['Y_merge'] = create_mergekey(incomes)
    jobs['X_merge'],jobs['Y_merge'] = create_mergekey(jobs)
    trees['X_merge'],trees['Y_merge'] = create_mergekey(trees)
    neighborhoods['X_merge'],neighborhoods['Y_merge'] = create_mergekey(neighborhoods)
    business_class02['X_merge'],business_class02['Y_merge'] = create_mergekey(business_class02)
    business_class07['X_merge'],business_class07['Y_merge'] = create_mergekey(business_class07)
    business_class08['X_merge'],business_class08['Y_merge'] = create_mergekey(business_class08)
    parks['X_merge'],parks['Y_merge'] = create_mergekey(parks)
    schools['X_merge'],schools['Y_merge'] = create_mergekey(schools)
    electricity['X_merge'],electricity['Y_merge'] = create_mergekey(electricity)
    employment['X_merge'],employment['Y_merge'] = create_mergekey(employment)
 
mergekeys()


# In[81]:

def merge_all():
    for i,df in enumerate([train_raw,test_raw]):
        t = 'train' if i==0 else 'test'
        
        logger.info('Merging income for {}'.format(t))
        logger.info('Pre datasize: {}'.format(len(df)))
        df = pd.merge(df,incomes[['income_id'                                   ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left',copy=False)

        logger.info('Merging jobs for {}'.format(t))
        df = pd.merge(df,jobs[['job_id'                                ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left')

        logger.info('Merging trees for {}'.format(t))
        df = pd.merge(df,trees[['trees'                                  ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left',copy=False)

        logger.info('Merging neighborhoods for {}'.format(t))
        df = pd.merge(df,neighborhoods[['neighborhood_id'                                        ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left',copy=False)

        logger.info('Merging parks for {}'.format(t))
        df = pd.merge(df,parks[['nearest_park_distance'                                ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left',copy=False)

        logger.info('Merging business class 02 for {}'.format(t))
        df = pd.merge(df,business_class02[                                        ['nearest_business_distance_02class'                                         ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left',copy=False)

        logger.info('Merging business class 07 for {}'.format(t))
        df = pd.merge(df,business_class07[['nearest_business_distance_07class'                                            ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left',copy=False)

        logger.info('Merging business class 08 for {}'.format(t))
        df = pd.merge(df,business_class08[['nearest_business_distance_08class'                                           ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left',copy=False)

        logger.info('Merging electricity for {}'.format(t))
        df = pd.merge(df,electricity[['electricity_id'                                      ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left',copy=False)

        logger.info('Merging employment for {}'.format(t))
        df = pd.merge(df,employment[['employment_id'                                      ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left',copy=False)

        logger.info('Merging schools for {}'.format(t))
        df = pd.merge(df,schools[['charter school nearby'                                    ,'county school nearby'                                    ,'elementary school nearby'                                    ,'high school nearby'                                    ,'middle school nearby'                                    ,'X_merge','Y_merge']]                             ,on=['X_merge','Y_merge'],how='left',copy=False)

        logger.info('Merging electricity props for {}'.format(t))
        df = pd.merge(df,props_df_elect,left_on='electricity_id'                                      ,right_on='Id'                                      ,how='left'                                      ,copy=False)

        logger.info('Merging job props for {}'.format(t))
        df = pd.merge(df,props_df_jobs,left_on='job_id'                                      ,right_on='Id'                                      ,how='left'                                      ,copy=False)

        logger.info('Merging income props for {}'.format(t))
        df = pd.merge(df,props_df_income,left_on='income_id'                                      ,right_on='Id'                                      ,how='left'                                      ,copy=False)

        logger.info('Merging employment props for {}'.format(t))
        df = pd.merge(df,props_df_employment,left_on='employment_id'                                      ,right_on='Id'                                      ,how='left'                                      ,copy=False)

        logger.info('Post datasize: {}'.format(len(df)))
        logger.info('Saving {} data to disk'.format(t))
        filename = 'Data/merge_{}_all.csv'.format(t)
        df.to_csv(filename)

merge_all()


# ### Part 4: Final model

# #### Prepare datasets for model training

# In[3]:

train_raw = pd.read_csv('Data/merge_{}_all.csv'.format('train'),index_col=0)
test_raw = pd.read_csv('Data/merge_{}_all.csv'.format('test'),index_col=0)


# In[4]:

model_fields = [#'Category',
                 'DayOfWeek',
                 'PdDistrict',
                 #'collar_id',
                 'DaySegment',
                 'TimeOfDay',
                 'DayOfMonth',
                 'Year',
                 'YearSegment',
                 'KMcluster',
                 'KMdistance',
                 'trees',
                 'neighborhood_id',
                 'nearest_park_distance',
                 'nearest_business_distance_02class',
                 'nearest_business_distance_07class',
                 'nearest_business_distance_08class',
                 'charter school nearby',
                 'county school nearby',
                 'elementary school nearby',
                 'high school nearby',
                 'middle school nearby',
                 'Pop2010_zc',
                 'Zip',
                 'kWh',
                 'kWh_pC',
                 'JOBS_CNT',
                 'JOBS_PSMI',
                 'MedInc_d',
                 'MedInc_moe',
                 'pC_Inc_d',
                 'pC_Inc_moe',
                 'Employ_moe',
                 'Employ_pct',
		 'AddressMod']
categorical_fields = [#'Category',
                 'DayOfWeek',
                 'PdDistrict',
                 #'collar_id',
                 'DaySegment',
                 'TimeOfDay',
                 'DayOfMonth',
                 'Year',
                 'YearSegment',
                 'KMcluster',
                 'neighborhood_id',
		 'AddressMod']


# In[5]:

logger.info('Sampling started')
tr = train_raw[model_fields+['collar_id','Category']].copy().iloc[                   np.random.permutation(len(train_raw))]

logger.info('Filling NA values')
numerics = [col for col in model_fields             if col not in categorical_fields]
tr[numerics] = tr[numerics].fillna(tr[numerics].mean())
tr.loc[tr['neighborhood_id']==-1,['neighborhood_id']] = np.nan
tr['neighborhood_id'] = tr['neighborhood_id'].fillna(99)

logger.info('Creating dev and test datasets')
dev_train    , dev_train_labels_cid    , dev_train_labels_cat = tr[model_fields][50001:],                             tr['collar_id'][50001:],                                tr['Category'][50001:]
dev_test    , dev_test_labels_cid    , dev_test_labels_cat = tr[model_fields][:50000],                            tr['collar_id'][:50000],                            tr['Category'][:50000]


# In[6]:

logger.info('Test data transform started')
logger.info('Filling NA values')
numerics = [col for col in model_fields             if col not in categorical_fields]
test_raw[numerics] = test_raw[numerics].fillna(test_raw[numerics].mean())
test_raw.loc[test_raw['neighborhood_id']==-1,['neighborhood_id']] = np.nan
test_raw['neighborhood_id'] = test_raw['neighborhood_id'].fillna(99)


# #### Make a prediction of crime metaclass and use the prediction & prediction score in final model

# In[7]:

def make_pipeline(model_fields,categorical_fields):
    ''' Create pipeline that will be 
        used multiple times.
        
        Pipeline contains all necessary
        components of model run in 
        sequence. 
        
        SVD runs prior to a RandomForest model
        and creates a dense metrix. RF model
        outputs new dataset with most important
        features to LogisticRegression model.
        
        The same logic will be used twice:
        1) predict collar ID and then add 
            the prediction and corresponding
            prediction probability as features
            on the original dataset
        2) predict Category and use for 
            the next submission.
    '''
    le = MultiColumnLabelEncoder()
    cf = [i for i,x in enumerate(model_fields) if x in categorical_fields]
    ohe = OneHotEncoder(categorical_features=cf,sparse=True)
    svd = TruncatedSVD(n_components=150) 
    ss = StandardScaler(copy=False)
    rf = RandomForestClassifier(n_estimators=70,n_jobs=1)
    lr = LogisticRegression(C=0.01,solver='lbfgs'                            ,multi_class='multinomial')
    pl = Pipeline([('le',le)        # Recode text features as integers with LabelEncoder
                   ,('ohe',ohe)     # Create dummy features for each categorical feature
                   ,('svd',svd)     # Decompose features into a smaller projection \
                                    #   as a dense matrix for scaler 
                   ,('ss',ss)       # Scale numerical variables 
                   ,('lr',lr)       # Transform data to most important features
                   ,('rf',rf)])     # Run Forest model on minimized dataset
                   
    return pl

logger.info('Creating pipeline')
pl = make_pipeline(model_fields,categorical_fields)


# In[8]:
cv = TfidfVectorizer()
svd = TruncatedSVD(n_components=100)
vect_address = Pipeline([('cv',cv),('svd',svd)])
x = vect_address.fit_transform(train_raw['AddressMod'])
dev_train = pd.concat([dev_train,pd.DataFrame(x,index=train_raw.index)],axis=1,join='inner') 
dev_train = dev_train.drop('AddressMod',1)

def predict_metaclass(pl):
    '''Predict the metaclass of crime
       Use the outputted algorithm to 
       predict metaclass in the training data
    '''
    
    logger.info('Fitting training data')   
    pl.fit(dev_train, dev_train_labels_cid)
    
    return pl

logger.info('Model for collar_id started')
pl_metaclass = predict_metaclass(pl)
logger.info('Model for collar_id ended')


# In[9]:

logger.info('Metaclass model diagnostics started')
dev_test = pd.concat([dev_test,pd.DataFrame(x,index=train_raw.index)],axis=1,join='inner')
dev_test = dev_test.drop('AddressMod',1)

logger.info('Model accuracy: {}%'.format(round(pl_metaclass.score(dev_test,                                              dev_test_labels_cid),4)*100))
predictions = pl_metaclass.predict(dev_test)
conf = confusion_matrix(dev_test_labels_cid, predictions)
plt.imshow(conf, cmap='binary', interpolation='nearest')
print pd.crosstab(dev_test_labels_cid, predictions,                   rownames=['True'], colnames=['Predicted'],                   margins=True)
print 'F1 Score: {}%'.format(round(f1_score(                    dev_test_labels_cid                    ,predictions                    ,average='weighted')*100,2))
print classification_report(dev_test_labels_cid                     ,predictions)

predictions = pl_metaclass.predict_proba(dev_test)
print log_loss(dev_test_labels_cid, predictions)


# #### Add prediction and prediction score as a feature
# 
# Use the generated dev and test datasets from above in future modeling efforts, because otherwise the model will have been fit with test data and accuracy measures will have been thrown off.

# In[10]:

cid_prediction = pl_metaclass.predict(dev_train)
cid_prediction_score = np.max(pl_metaclass.predict_proba(dev_train),axis=1)
dev_train['cid_prediction'] = cid_prediction
dev_train['cid_pred_score'] = cid_prediction_score 

cid_prediction = pl_metaclass.predict(dev_test)
cid_prediction_score = np.max(pl_metaclass.predict_proba(dev_test),axis=1)
dev_test['cid_prediction'] = cid_prediction
dev_test['cid_pred_score'] = cid_prediction_score 


# In[11]:

x = vect_address.transform(test_raw['AddressMod'])
test_raw = pd.concat([test_raw[model_fields],pd.DataFrame(x,index=test_raw.index)],axis=1,join='inner')
test_raw = test_raw.drop('AddressMod',1)

cid_prediction = pl_metaclass.predict(test_raw)
cid_prediction_score = np.max(pl_metaclass.predict_proba(test_raw),axis=1)
test_raw['cid_prediction'] = cid_prediction
test_raw['cid_pred_score'] = cid_prediction_score


# In[12]:

del pl_metaclass


# #### Make final model of crime category and test performance

# In[13]:

categorical_fields = categorical_fields + ['cid_prediction']
pl = make_pipeline(model_fields,categorical_fields)


# In[14]:
def make_pipeline():
    ''' Create model that will be used for final
        prediction of category. Will use
        Neural Networks after some transformation.
    '''

    ''' Compile pipeline transformers '''
    le = MultiColumnLabelEncoder()
    cf = [i for i,x in enumerate(model_fields) if x in categorical_fields]
    ohe = OneHotEncoder(categorical_features=cf,sparse=True)
    svd = TruncatedSVD(n_components=150)
    ss = StandardScaler()
    lr = LogisticRegression(C=0.01,solver='lbfgs',multi_class='multinomial')

    ''' Create pipeline '''
    logger.info('Compiling pipeline')
    pl = Pipeline([('le',le)        # Recode text features as integers with LabelEncoder
                   ,('ohe',ohe)     # Create dummy features for each categorical feature
                   ,('svd',svd)     # Decompose features into a smaller projection \
		   ,('ss',ss)       # Scale features before LR
                   ,('lr',lr)])     # Run LR model on minimized dataset

    return pl

pl = make_pipeline()

def predict_category(pl):
    '''Predict the category of crime
        and use for submission.
    '''

    logger.info('Fitting training data')
    lb = LabelBinarizer()
    tr = pl.fit_transform(dev_train, dev_train_labels_cat)
    
    ''' Compile NN '''
    logger.info('Compiling neural network')
    n = tr.shape[1]
    nn = Sequential()
    nn.add(GaussianDropout(0.5))
    nn.add(Dense(input_dim=n, output_dim=64, init="uniform"))
    nn.add(Activation("tanh"))
    nn.add(PReLU((64,)))
    nn.add(Dropout(0.3))
    nn.add(Dense(input_dim=64, output_dim=39, init="glorot_uniform"))
    nn.add(Activation("softmax"))
    nn.compile(loss='categorical_crossentropy', optimizer='sgd')
    
    cat_train = lb.fit_transform(dev_train_labels_cat)
    nn.fit(tr, cat_train)
    return pl,lb,nn

logger.info('Model for Category started')
pl_category,lb_category,nn_category = predict_category(pl)
logger.info('Model for Category ended')


logger.info('Model diagnostics started')
cat_test = lb_category.transform(dev_test_labels_cat)
tr = pl_category.transform(dev_test)
predictions = nn_category.predict(tr)
predictions_t = lb_category.inverse_transform(predictions)
logger.info('Model accuracy: {}%'.format(round(np.mean(dev_test_labels_cat==predictions_t),4)*100))
conf = confusion_matrix(dev_test_labels_cat, predictions_t)
plt.imshow(conf, cmap='binary',interpolation='nearest')
print pd.crosstab(dev_test_labels_cat, predictions_t,                   rownames=['True'], colnames=['Predicted'],                   margins=True)
print 'F1 Score: {}%'.format(round(f1_score(                    dev_test_labels_cat                     ,predictions_t                     ,average='weighted')*100,2))
print classification_report(dev_test_labels_cat                     ,predictions_t)

predictions = nn_category.predict_proba(tr)
try:
	print log_loss(dev_test_labels_cat, predictions)
except:
	pass

# #### Make final submission


del dev_train
del dev_test
del dev_test_labels_cat
del dev_test_labels_cid

tr = pl_category.transform(test_raw)
predictions = nn_category.predict_proba(tr)

def write_to_submissionfile():
    cols = set(train_raw['Category'])
    submission_df = pd.DataFrame(predictions,columns=sorted(cols))
    submission_df.rename(columns={'SEX OFFENSES, FORCIBLE':'SEX OFFENSES FORCIBLE'}, inplace=True)
    submission_df.rename(columns={'SEX OFFENSES, NON FORCIBLE':'SEX OFFENSES NON FORCIBLE'}, inplace=True)
    submission_df.to_csv('Data/submission_file_final.csv',index_label='Id',na_rep=0)
    
write_to_submissionfile()

