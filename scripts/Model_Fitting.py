# -*- coding: utf-8 -*-
"""
The big boy script which contains all the model fitting. This should be the last file run.
"""

import os

root_dir = 'Z:\\GDP\\GDP_prediction_minimal_implementation\\'
os.chdir(root_dir)

import numpy as np
import pandas as pd
from utils.CountryIterator import CountryIterator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(57)

long_locs_path = root_dir+"full_sample.csv"
gdp_path = "D:\\HomeWorking\\predicting-poverty-replication-master\\junk\\predicting-poverty-replication-master\\GDP\\GDP\\GDP_Data.csv" #This path is where the GDP data csv should be. I've included the file I used as a template.
country_list_file = "D:\\HomeWorking\\predicting-poverty-replication-master\\junk\\predicting-poverty-replication-master\\GDP\\countries.csv"

n_dim = 5
n_samp = 1000

country_codes = CountryIterator.get_country_codes(country_list_file,exclusions=["Kiribati"])
countries = country_codes['country']

df = pd.read_csv(long_locs_path, sep=',')

def weighted_avg_and_std(values, weights): #https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

df.drop([col for col in df.columns.values if "Unnamed" in col],axis=1,inplace=True) #A load of garbage columns accumulated over time.

mus = []
stds = [] #As a teenager, I asked my dad what 'std' stood for and he, as a statistician, responded 'standard deviation'.
for i in range(0,n_dim):
    mu, std = weighted_avg_and_std(df['N'+str(i)],df['weight'])
    mus.append(mu)
    stds.append(std)
    df['nN'+str(i)] = (df['N'+str(i)]-mu)/std #Normalizing.

df.insert(df.shape[1],'w_rad',df['weight']*df['radiance']) #Weighted radiance.

#Processing the gdp data.
dg = df.groupby(['country']).agg('sum') #This is the gdp dataframe. I should rename this.
gdp = pd.read_csv(gdp_path, encoding='cp1252')
gdp = gdp.loc[~((gdp['2019 [YR2019]']=='..') | (gdp['Country Name'].isnull()))]
gdp.insert(gdp.shape[1],'country_code', gdp['Country Code'])
temp_table = gdp.join(country_codes.set_index('country_code'), on='country_code', how='inner').join(dg,on='country',how='inner',rsuffix='_')
dg = temp_table[['country','2019 [YR2019]']].join(dg,on='country',how='right')
dg['GDP19'] = dg['2019 [YR2019]'].astype(float)
inds = ~np.isnan(dg['GDP19'])
dg['log_w_rad'] = np.log10(dg['w_rad'])

  
np.random.seed(9490)
validation_countries = np.random.choice(dg.loc[inds,'country'], int(len(dg.loc[inds,'country'])/5),replace=False)
train_countries = dg.loc[inds,'country']
train_countries = train_countries[~train_countries.isin(validation_countries)]
dft = df.loc[df.country.isin(train_countries)].copy() #Training sample points,
dfv = df.loc[df.country.isin(validation_countries)].copy() #validation sample points.
dgt = dg.loc[dg.country.isin(train_countries)].copy() #Training GDP observations.
dgv = dg.loc[dg.country.isin(validation_countries)].copy() #Validation GDP observations.

#This expression kept cropping up a lot in code so I made it a function.
def F(x):
    return x.to_numpy().reshape(-1,1)

#In the paper we restricted v (i.e. s) to be a unit vector.
#This was a later convention we adopted for convenience.
#Here in the code, v (i.e. s) has length, t_1 = 1, and the input variable t actually defines the difference between the successive terms.
#This function is a utility function which takes the weird convention we use here in the code and returns the same form you see in the paper.
def toNormalForm(s,t):
    t0 = np.linalg.norm(s)**2
    m = len(t)+2
    T = np.empty(len(t)+1)
    T[0] = t0
    for i in range(1,m-1):
        T[i] = T[i-1] + t[i-1]
    T.sort()
    l = np.linalg.norm(s)
    s = s/l
    T = T*l
    return (s,T)


def get_model_CVMSE(X,Y, k_folds=5):
    ordering = np.tile(range(0,k_folds),int(1 + len(Y)/k_folds))[0:len(Y)]
    np.random.shuffle(ordering)
    score = 0
    for i in range(0,k_folds):
        regress2 = LinearRegression().fit(X[~(ordering==i)],Y[~(ordering==i)])
        score += mean_squared_error(regress2.predict(X[ordering==i]),Y[ordering==i]) / k_folds
    return score
    

#Handles all the fitting logic.
#Note that t=None just gives a regular nightlights regression.
#See Section III.I from the paper.
def get_model_score(df, s, t=None):
    t0 = np.linalg.norm(s)**2
    df['cluster_dash'] = 0
    df['proj'] = df[['nN0','nN1','nN2','nN3','nN4']].dot(s) #Numpy is so elegant at times.
    if t is not None:
        m = len(t)+2
        T = np.empty(len(t)+1)
        T[0] = t0
        for i in range(1,m-1):
            T[i] = T[i-1] + t[i-1]
        T.sort()
        for Ti in T:
            df['cluster_dash'] += (df['proj'] > Ti)*1
    else:
        m = 2
        df['cluster_dash'] = (df['proj'] > t0)*1
    cols=[]
    for i in range(0,m):
        col = 'r'+str(i)+'_dash'
        df[col] = df['radiance']*df['weight']*(df['cluster_dash']==i)
        cols.append(col)
    dgg = df[['country']+cols].groupby(['country']).agg('sum')
    dgg = temp_table[['country','2019 [YR2019]']].join(dgg,on='country',how='right')
    dgg['GDP19'] = dgg['2019 [YR2019]'].astype(float)
    inds = ~np.isnan(dgg['GDP19'])
    X = dgg.loc[inds,cols]
    X = np.log10(X+1)
    Y = np.log10(dgg.loc[inds,'GDP19'])
    return get_model_CVMSE(X,Y)


#A Monte Carlo search to find the optimal s,t values.
#New values it tests are actually often generated near the previous best result so it's sort of like a genetic algorithm.
#Epsilon determines the percentage of the time a point is generated near the best result. 
#Epsilon=1 means this never happens, epsilon=0 means it always happens.
#Sigma determines the variance of the distribution of the normal distribution around the best point when that step occurs.
#If decay is true, sigma decays by a factor of beta every step.
#This decay is good for honing in on the optimal value.
#n_samp is the number os search steps to perform.
#The number of regions p will stay constant throughout the search which will be the length of the input t + 2
def model_search_fast(df, n_samp = 100, seed=4550, epsilon=0.8, sigma=0.05, s=None, t=[1,2], decay=False, beta=0.999):
    results = ([],[],[])
    np.random.seed(seed)
    rand = 0
    if s is not None:
        best_s = s
    else:
        best_s = np.random.normal(0,sigma*5,size=n_dim)
    best_t = t
    minner = get_model_score(df,best_s,best_t)
    for i in tqdm(range(0,n_samp)):
        if rand < epsilon:
            s = np.random.normal(0,0.5,size=n_dim)
            t = np.random.normal(1,0.5,size=len(t))
        else:
            s = best_s + np.random.normal(0,sigma,size=n_dim)
            t = best_t + np.random.normal(0,sigma,size=len(t))
        LL = get_model_score(df,s,t)
        results[0].append(s)
        results[1].append(t)
        results[2].append(LL)
        if LL < minner:
            print(t)
            print(s)
            print(LL)
            minner = LL
            best_s = s
            best_t = t
        if decay:
            sigma *= beta
    return results


#This is needed for the plots.
#dfv is for the validation set.
#model is for if you just want to test a model without refitting.
def get_model_R2(df, s, t, dfv=None, model=None):
    t0 = np.linalg.norm(s)**2
    df['cluster_dash'] = 0
    df['proj'] = df[['nN0','nN1','nN2','nN3','nN4']].dot(s)
    if t is not None:
        m = len(t)+2
        T = np.empty(len(t)+1)
        T[0] = t0
        for i in range(1,m-1):
            T[i] = T[i-1] + t[i-1]
        T.sort()
        for Ti in T:
            df['cluster_dash'] += (df['proj'] > Ti)*1
    else:
        m = 2
        df['cluster_dash'] = (df['proj'] > t0)*1
    cols=[]
    for i in range(0,m):
        col = 'r'+str(i)+'_dash'
        df[col] = df['radiance']*df['weight']*(df['cluster_dash']==i)
        cols.append(col)
    dgg = df[['country']+cols].groupby(['country']).agg('sum')
    dgg = temp_table[['country','2019 [YR2019]']].join(dgg,on='country',how='right')
    dgg['GDP19'] = dgg['2019 [YR2019]'].astype(float)
    inds = ~np.isnan(dgg['GDP19'])
    X = dgg.loc[inds,cols]
    X = np.log10(X+1)
    Y = np.log10(dgg.loc[inds,'GDP19'])
    print("Entering.")
    if model is not None:
        print("Model is not None.")
        return model.score(X,Y)
    print(len(X))
    model = LinearRegression().fit(X,Y)
    if dfv is not None: #After fitting, the calculation needs to go through the above steps again for the validation set..
        print("dfv is not None.")
        return get_model_R2(dfv,s,t,model=model)
    else:
        print("dfv is None.")
        return model.score(X,Y)


#Outputs a model for a given s and t.
def get_model(s,t,df=df):
    t0 = np.linalg.norm(s)**2
    df['cluster_dash'] = 0
    df['proj'] = df[['nN0','nN1','nN2','nN3','nN4']].dot(s)
    if t is not None:
        m = len(t)+2
        T = np.empty(len(t)+1)
        T[0] = t0
        for i in range(1,m-1):
            T[i] = T[i-1] + t[i-1]
        T.sort()
        for Ti in T:
            df['cluster_dash'] += (df['proj'] > Ti)*1
    else:
        m = 2
        df['cluster_dash'] = (df['proj'] > t0)*1
    cols=[]
    for i in range(0,m):
        col = 'r'+str(i)+'_dash'
        df[col] = df['radiance']*df['weight']*(df['cluster_dash']==i)
        cols.append(col)
    dgg = df[['country']+cols].groupby(['country']).agg('sum')
    dgg = temp_table[['country','2019 [YR2019]']].join(dgg,on='country',how='right')
    dgg['GDP19'] = dgg['2019 [YR2019]'].astype(float)
    inds = ~np.isnan(dgg['GDP19'])
    X = dgg.loc[inds,cols]
    X = np.log10(X+1)
    Y = np.log10(dgg.loc[inds,'GDP19'])
    regress2 = LinearRegression().fit(X,Y)
    return regress2


optimal_s = [ 0.46118698,  0.004349, -0.76801022, -0.07657351, -0.71737465]
optimal_t = [-1.53579077, 1.23357487]
toNormalForm(optimal_s,optimal_t)

get_model_score(dft,optimal_s,optimal_t)
get_model_R2(df,optimal_s,[])
get_model_R2(dft,optimal_s,optimal_t,dfv) #Validation R^2.
get_model_R2(df,optimal_s,optimal_t) #Final R^2

results = model_search_fast(dft,5000,sigma=1,epsilon=0,t=optimal_t, s=optimal_s, decay=True,beta=0.997)


def make_scatters(s,t): #Produces Fig. 4 from the paper.
    get_model(s=s,t=t,df=dfv)
    test_model = get_model(s=s,t=t,df=dft)
    
    
    cols = ['r'+str(i)+'_dash' for i in range(0,len(t)+2)]
    
    final_model = get_model(s=s,t=t,df=df)
    
    #Calculates the predictions for the whole study area using the final model.
    dgg = df[['country']+cols].groupby(['country']).agg('sum')
    dgg = temp_table[['country','2019 [YR2019]','Country Code']].join(dgg,on='country',how='right')
    dgg['GDP19'] = dgg['2019 [YR2019]'].astype(float)
    inds = ~np.isnan(dgg['GDP19'])
    X = dgg.loc[inds,cols]
    X = np.log10(X+1)
    Y = np.log10(dgg.loc[inds,'GDP19'])
    
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(Y,Y-final_model.predict(X),s=5,alpha=0.7,edgecolors='None')
    ax.set(xlabel="log10 GDP",ylabel="log10 GDP - log10 pred GDP")
    lims=ax.get_xlim()
    ax.plot(lims, [0,0], 'r--', alpha=0.75, zorder=0, linewidth=1)
    ax.set_xlim(lims)
    l = np.max(np.abs(ax.get_ylim()))
    ax.set_ylim([-l,l])
    ax.set_aspect('equal')
    plt.savefig("residuals.png",dpi=300)
    
    plt.clf()
    fig, ax = plt.subplots(1,2)
    
    ax[1].scatter(final_model.predict(X),Y,s=5,alpha=0.7,edgecolors='None')
    ax[1].annotate("R^2 = {:.3f}".format(np.round(final_model.score(X,Y),3)), (8,13), fontsize=8)
    lims = [
        np.min([ax[1].get_xlim(), ax[1].get_ylim()]),  # min of both axes
        np.max([ax[1].get_xlim(), ax[1].get_ylim()]),  # max of both axes
    ]

    ax[1].set_xlim(lims)
    ax[1].set(xlabel="log10 Predicted GDP")
    ax[1].plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=1)
    ax[1].set_aspect('equal')
    ax[1].set_ylim(lims)
    
    #This does the calculations for the validation set using the model fitted on the training set.
    dgg = dfv[['country']+cols].groupby(['country']).agg('sum')
    dgg = temp_table[['country','2019 [YR2019]']].join(dgg,on='country',how='right')
    dgg['GDP19'] = dgg['2019 [YR2019]'].astype(float)
    inds = ~np.isnan(dgg['GDP19'])
    X = dgg.loc[inds,cols]
    X = np.log10(X+1)
    Y = np.log10(dgg.loc[inds,'GDP19'])
    
    
    ax[0].scatter(test_model.predict(X),Y,s=5,alpha=0.7,edgecolors='None')
    ax[0].annotate("R^2 = {:.3f}".format(np.round(test_model.score(X,Y),3)), (8,13), fontsize=8)

    ax[0].set_xlim(lims)
    #ax[0].set_title("Predicted GDP vs actual GDP on validation set")
    ax[0].set(ylabel="log10 GDP",xlabel="log10 Predicted GDP")
    ax[0].plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=1)
    ax[0].set_aspect('equal')
    ax[0].set_ylim(lims)
    
    plt.savefig("scatters_hyperplane.eps",format='eps')
    
make_scatters(optimal_s,optimal_t)

def make_map_visual(s,t): #This recreates Fig. 2 from the paper.
    import descartes
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    import matplotlib.lines as mlines
    print(get_model_R2(df,s,t))
    country_df = df.loc[df.country=="Tanzania"]
    mapp = gpd.read_file('D:\\HomeWorking\\tza_admbnda_adm0_20181019').to_crs('epsg:4326') #Ew
    geo_df = [Point(xy) for xy in zip(country_df['lons'],country_df['lats'])]
    geo_df = gpd.GeoDataFrame(country_df, crs={'init':'epsg:4326'},geometry=geo_df)
    fig,ax=plt.subplots(1,2)
    ax[0].axis('off')
    mapp.plot(ax=ax[0],color='black',alpha=0.8)
    geo_df.plot(ax=ax[0],alpha=0.5,markersize=2,edgecolors='none',c='yellow')
    ax[1].axis('off')
    mapp.plot(ax=ax[1],color='black',alpha=0.8)
    cmap = ListedColormap(['red','yellow','cornflowerblue','lime'], name='custard')
    geo_df.plot(ax=ax[1],column=country_df['cluster_dash'],alpha=1,markersize=2,edgecolors='none',cmap=cmap)
    plt.savefig("UK_sampleHyper.png",dpi=500)
    
make_map_visual(optimal_s,optimal_t)

### BOOTSTRAP VARIANCE ###
#This estimates the sampling error. It's just the variance with no bias because the estimator is unbiased.
model = get_model(s=optimal_s,t=optimal_t,df=df)
cols = ['r'+str(i)+'_dash' for i in range(0,4)]
replications = 50
country_ests = np.empty(countries.shape[0])
for j in range(0,countries.shape[0]):
    country = countries[j]
    ests = np.empty(replications)
    for i in tqdm(range(0,replications)):
        country_df = df.loc[df.country==country,['country']+cols]
        samp = country_df.sample(n_samp,replace=True) #Selecting with equal probabiltites from the sample emulates the PPS sampling scheme as the sample is already PPS representative.
        X = samp.groupby('country').agg('sum')
        X = np.log10(X+1)
        ests[i] = model.predict(X)[0]
    country_ests[j] = np.sqrt(ests.var())/ests.mean()
    print(country+" complete.")
    print(country_ests[j]*100)
        
        
        