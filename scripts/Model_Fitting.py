# -*- coding: utf-8 -*-
"""
The big boy script which contains all the model fitting. This should be the last file run.
"""

import os

root_dir = 'H:\\GDP\\GDP_prediction_minimal_implementation\\'
os.chdir(root_dir)

import numpy as np
import pandas as pd
from utils.CountryIterator import CountryIterator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

np.random.seed(57)

long_locs_path = root_dir+"full_sample.csv"
gdp_path = "H:\\GDP\\GDP\\GDP_Data.csv" #This path is where the GDP data csv should be. I've included the file I used as a template.
country_list_file = "H:\\GDP\\countries.csv"

n_dim = 12
n_samp = 1000

country_codes = CountryIterator.get_country_codes(country_list_file,exclusions=["Kiribati"])
countries = country_codes['country']

df = pd.read_csv(long_locs_path, sep=',')

df2 = pd.read_csv(root_dir+"image_features2.csv")
for i in range(0,n_dim):
    df['N'+str(i)] = df2['N'+str(i)]

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
    cols = ['nN'+str(i) for i in range(0,len(s))]
    df['proj'] = df[cols].dot(s) #Numpy is so elegant at times.
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
def model_search_fast(df, n_samp = 100, seed=4550, epsilon=0.8, sigma=0.05, s=None, t=[1,2], decay=False, beta=0.999, min_region_size=256):
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
            s = best_s + np.random.normal(0,sigma,size=len(s))
            t = best_t + np.random.normal(0,sigma,size=len(t))
        LL = get_model_score(df,s,t)
        results[0].append(s)
        results[1].append(t)
        results[2].append(LL)
        if LL < minner:
            if df.groupby('cluster_dash').agg('count')['im_name'].min() >= min_region_size:
                print(t)
                print(s)
                print(LL)
                minner = LL
                best_s = s
                best_t = t
        if decay:
            sigma *= beta
    return best_s, best_t, minner


#This is needed for the plots.
#dfv is for the validation set.
#model is for if you just want to test a model without refitting.
def get_model_R2(df, s, t, dfv=None, model=None):
    t0 = np.linalg.norm(s)**2
    df['cluster_dash'] = 0
    cols = ['nN'+str(i) for i in range(0,len(s))]
    df['proj'] = df[cols].dot(s)
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
    cols = ['nN'+str(i) for i in range(0,len(s))]
    df['proj'] = df[cols].dot(s)
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

def loop_results(df,init_s,init_t=[],max_regions=5,iters=1000,stopping_value=1e-2,min_region_size=512): #Get log-likelihood for increasing number of regions.
    Ss = []
    Ts = []
    LLs = []
    s = init_s
    t = init_t.copy()
    for i in range(len(init_t)+2,max_regions+1):
        beta = np.power(stopping_value,1/iters)
        results = model_search_fast(df,iters,sigma=1,epsilon=0,t=t,s=s,decay=True,beta=beta,min_region_size=min_region_size)
        s = results[0]
        t = results[1]
        Ss.append(s)
        Ts.append(t)
        LLs.append(results[2])
        t = np.append(t,0.01)
    return Ss, Ts, LLs

def compute_AICs(df,init_s=[0.5,0.5,0.5],max_dim=7,max_regions=5,iters=1000,stopping_value=1e-2,min_region_size=512):
    Ss = []
    Ts = []
    LLs = []
    AICs = []
    train_n = train_countries.shape[0]
    init_s = init_s.copy()
    for n_dim in range(len(init_s)+1,max_dim+1):
        init_s.append(0.5)
        a,b,c = loop_results(df,init_s,init_t=[],max_regions=max_regions,iters=iters,stopping_value=stopping_value,min_region_size=min_region_size)
        Ss = Ss+a
        Ts = Ts+b
        LLs = LLs+c
        for i in range(0,len(c)):
            k = len(init_s)+2*len(b[i])+2+2
            k_dash = k+(k**2 + k)/(train_n-k+1)
            ll = np.log(c[i])*train_n
            aic = ll + 2*k_dash
            AICs.append(aic)
            print("n_dim = {}, splits = {}, AICc = {}".format(str(n_dim),str(b[i]+1),str(aic)))
    return Ss, Ts, AICs
    
            
n_seeds=2
aic_iters=500
d, p, AICs = [None,None,None]
vR2s = []
R2s = []
for seed in range(1,n_seeds):
    np.random.seed(seed)
    validation_countries = np.random.choice(dg.loc[inds,'country'], int(len(dg.loc[inds,'country'])/5),replace=False)
    train_countries = dg.loc[inds,'country']
    train_countries = train_countries[~train_countries.isin(validation_countries)]
    dft = df.loc[df.country.isin(train_countries)].copy() #Training sample points,
    dfv = df.loc[df.country.isin(validation_countries)].copy() #validation sample points.
    dgt = dg.loc[dg.country.isin(train_countries)].copy() #Training GDP observations.
    dgv = dg.loc[dg.country.isin(validation_countries)].copy() #Validation GDP observations.
    ss,ts,aics = compute_AICs(dft,iters=aic_iters)
    ind = np.argmin(aics)
    s = ss[ind]
    t = ts[ind]
    s,t,_ = model_search_fast(dft,n_samp=aic_iters*10,sigma=1,epsilon=0.05,t=t,s=s,decay=True,beta=(1e-2)**(1/aic_iters/10),min_region_size=512)
    vR2s.append(get_model_R2(dft,s,t,dfv=dfv))
    R2s.append(get_model_R2(df,s,t))
    if d is None:
        d = [len(s) for s in ss]
        p = [len(t)+2 for t in ts]
        AICs = [[aic] for aic in aics]
    else:
        for i in range(len(aics)):
            AICs[i].append(aics[i])
    print("Seed " + str(seed) + " complete :)")
    
aic_iters=1000
np.random.seed(999)
train_countries = dg.loc[inds,'country']
ss,ts,aics = compute_AICs(df,iters=aic_iters)
ind = np.argmin(aics)
s = ss[ind]
t = ts[ind]
s,t,_ = model_search_fast(df,n_samp=aic_iters*10,sigma=1,epsilon=0.05,t=t,s=s,decay=True,beta=(1e-2)**(1/aic_iters/10),min_region_size=512)

s = np.array([-0.18966549,  0.2621466 ,  0.38608642,  0.05394268,  0.42624443])
t = np.array([-0.14409082])
    

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
    
    plt.savefig("scatters_hyperplane.eps",dpi=300)
    
make_scatters(optimal_s,optimal_t)

def make_map_visual(s,t): #This recreates Fig. 2 from the paper.
    import descartes
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    import matplotlib.lines as mlines
    print(get_model_R2(df,s,t))
    country_df = df.loc[df.country=="Sweden"]
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
    
make_map_visual(s,t)

get_model_R2(df,s,t)

np.random.seed(5)
sample_countries = np.random.choice(dg.loc[inds,'country'], 100,replace=False)
sampdf = df.loc[df.country.isin(sample_countries)]
sample_df = sampdf[sampdf.cluster_dash==0].sample(n=150)
for c in range(1,3):
    sample_df = sample_df.append(sampdf[sampdf.cluster_dash==c].sample(n=150))

from PIL import Image
images_main_path = r'H:\GDP\GDP_prediction_minimal_implementation\sample3'   
sample_df['type']=''
def mass_open(df):
    for _, r in df.iterrows():
        Image.open(os.path.join(images_main_path,r.im_name)).show()
def start_labelling(df,col=None,value=None):
    inp = ''
    while not inp=='quit':
        if col is not None and value is not None:
            i = np.random.choice(df[df[col]==value].index)
        else:
            i = np.random.choice(df[df['type']==''].index)
        mass_open(df.loc[[i]])
        inp = input('Land type:')
        if inp=='quit':
            continue
        if inp=="dense urban":
            print(np.sum(df.type==''))
        df.at[i,'type'] = inp
        
start_labelling(sample_df)

### BOOTSTRAP VARIANCE ###
#This estimates the sampling error. It's just the variance with no bias because the estimator is unbiased.
model = get_model(s=s,t=t,df=df)
cols = ['r'+str(i)+'_dash' for i in range(0,3)]
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
        
def test_error_against_country_size(s=s,t=t,df=df,i=0):
    cols = ['r'+str(i)+'_dash' for i in range(0,len(t)+2)]
    model = get_model(s,t)
    dgg = df[['country']+cols].groupby(['country']).agg('sum')
    dgg = temp_table[['country','2019 [YR2019]','area']].join(dgg,on='country',how='right')
    dgg['GDP19'] = dgg['2019 [YR2019]'].astype(float)
    inds = ~np.isnan(dgg['GDP19'])
    X = dgg.loc[inds,cols]
    X = np.log10(X+1)
    Y = np.log10(dgg.loc[inds,'GDP19'])
    Y2 = np.log10(dgg.loc[inds,'area'])
    print(X.shape)
    r = np.corrcoef(X.iloc[:,i],Y)[0,1]
    n = Y.shape[0]
    t = r*np.sqrt(n-2)/np.sqrt(1-r**2)
    print("r="+str(r))
    print("n="+str(n))
    print("t="+str(t))
    return np.corrcoef(Y2,Y)

def get_preds(df,s,t):
    cols = ['r'+str(i)+'_dash' for i in range(0,len(t)+2)]
    model = get_model(s,t)
    dgg = df[['country']+cols].groupby(['country']).agg('sum')
    dgg = temp_table[['country','2019 [YR2019]','area']].join(dgg,on='country',how='right')
    dgg['GDP19'] = dgg['2019 [YR2019]'].astype(float)
    inds = ~np.isnan(dgg['GDP19'])
    X = dgg.loc[inds,cols]
    X = np.log10(X+1)
    Y = np.log10(dgg.loc[inds,'GDP19'])
    Ydash = model.predict(X)
    dgg = dgg[inds]
    dgg['Ydash'] = Ydash
    dgg['diff'] = Y - Ydash
    dgg['Y'] = Y
    return dgg


### min_region_size hyperparameter search ###
   
    
def get_model_score(df, s, T=None):
    t0 = np.linalg.norm(s)**2
    df['cluster_dash'] = 0
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
    dgg = temp_table[['country','Y']].join(dgg,on='country',how='right')
    X = dgg[cols]
    X = np.log10(X+1)
    Y = dgg['Y']
    return get_model_CVMSE(X,Y)

def model_search_fast(df, n_samp = 100, seed=4550, epsilon=0.8, sigma=0.05, s=None, T=[1,2], decay=False, beta=0.999, min_region_size=256):
    results = ([],[],[])
    np.random.seed(seed)
    rand = 0
    best_t = T
    minner = get_model_score(df,s,best_t)
    for i in tqdm(range(0,n_samp)):
        if rand < epsilon:
            T = np.random.normal(1,1,size=len(T))
        else:
            T = best_t + np.random.normal(0,sigma,size=len(T))
        LL = get_model_score(df,s,T)
        if LL < minner:
            if df.groupby('cluster_dash').agg('count')['im_name'].min() >= min_region_size:
                print(T)
                print(s)
                print(LL)
                minner = LL
                best_s = s
                best_t = T
        if decay:
            sigma *= beta
    return best_s,best_t,minner
    
def get_model_R2(df, s, T, dfv=None, model=None):
    t0 = np.linalg.norm(s)**2
    df['cluster_dash'] = 0
    cols = ['nN'+str(i) for i in range(0,len(s))]
    df['proj'] = df[cols].dot(s)
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
    dgg = temp_table[['country','Y']].join(dgg,on='country',how='right')
    X = dgg[cols]
    X = np.log10(X+1)
    Y = dgg['Y']
    print("Entering.")
    if model is not None:
        print("Model is not None.")
        return model.score(X,Y)
    print(len(X))
    model = LinearRegression().fit(X,Y)
    if dfv is not None: #After fitting, the calculation needs to go through the above steps again for the validation set..
        print("dfv is not None.")
        return get_model_R2(dfv,s,T,model=model)
    else:
        print("dfv is None.")
        return model.score(X,Y)

#min_region_size hyperparameter search
def swagglejaggle(df,min_region_size,replications=5,noise=0,iters=5000,acc=1e-4):
    beta = 1e-4
    beta = np.power(beta,1/iters)
    scores = []
    for i in range(0,replications):
        for k in range(2,5):
            n_regions = 2*k
            s = np.random.normal(0,1,size=n_dim)
            s /= np.linalg.norm(s) 
            cols = ['nN'+str(i) for i in range(0,len(s))]
            df['proj'] = df[cols].dot(s) #Numpy is so elegant at times
            T = np.random.normal(0,1,size=n_regions-1)
            T.sort()
            df['cluster_dash'] = 0
            for Ti in T:
                df['cluster_dash'] += (df['proj'] > Ti)*1
            cols=[]
            for i in range(0,n_regions):
                col = 'r'+str(i)+'_dash'
                df[col] = df['radiance']*df['weight']*(df['cluster_dash']==i)
                cols.append(col)
            alpha = np.random.uniform(-0.3,1.5,size=n_regions)
            dgg = df[['country']+cols].groupby(['country']).agg('sum')
            dgg = temp_table[['country']].join(dgg,on='country',how='left')
            X = dgg[cols]
            X = np.log10(X+1)
            Y = X.dot(alpha)+np.random.normal(1,1)
            Y = Y.values + np.random.normal(0,noise,len(Y))
            temp_table['Y'] = Y
            validation_countries = np.random.choice(dg.loc[inds,'country'], int(len(dg.loc[inds,'country'])/5),replace=False)
            train_countries = dg.loc[inds,'country']
            train_countries = train_countries[~train_countries.isin(validation_countries)]
            dft = df.loc[df.country.isin(train_countries)].copy() #Training sample points,
            dfv = df.loc[df.country.isin(validation_countries)].copy() #validation sample points.
            T = np.random.normal(0,1,6)
            s,t,ll = model_search_fast(dft,iters,sigma=1,epsilon=0,T=T,s=s,decay=True,beta=beta,min_region_size=min_region_size,seed=np.random.randint(0,999999))
            scores.append(get_model_R2(dft,s,t,dfv))
    return np.mean(scores)
 
np.random.seed(569765)
means = []           
for k in range(4,10):
    means.append(swagglejaggle(df,min_region_size=2**k,replications=40,noise=1,iters=1500,acc=1e-4))
    print("Mean R^2 for min_region_size of "+str(2**k)+" is "+str(means[len(means)-1]))
print(means)
#[0.8386207610915373,0.8296783509081149,0.817977912602251,0.8137390940477095,0.8133585118289764,0.8492903746672145]


###################################
######## Subnational study ########
###################################

def regionalize(dfsub,s,t=None):
    t0 = np.linalg.norm(s)**2
    dfsub['cluster_dash'] = 0
    cols = ['nN'+str(i) for i in range(0,len(s))]
    dfsub['proj'] = dfsub[cols].dot(s) #Numpy is so elegant at times.
    if t is not None:
        m = len(t)+2
        T = np.empty(len(t)+1)
        T[0] = t0
        for i in range(1,m-1):
            T[i] = T[i-1] + t[i-1]
        T.sort()
        for Ti in T:
            dfsub['cluster_dash'] += (dfsub['proj'] > Ti)*1
    else:
        m = 2
        dfsub['cluster_dash'] = (dfsub['proj'] > t0)*1
    return m


gdpsub_path = "H:\\GDP\\GDP\\UScountiesGDP2020.csv"

dfsub = pd.read_csv(root_dir+"full_sample_sub_2.csv")

dfsub.drop([col for col in dfsub.columns.values if "Unnamed" in col],axis=1,inplace=True) #A load of garbage columns accumulated over time.

mus = []
stds = [] #As a teenager, I asked my dad what 'std' stood for and he, as a statistician, responded 'standard deviation'.
for i in range(0,8):
    mu, std = weighted_avg_and_std(dfsub['N'+str(i)],dfsub['weight'])
    mus.append(mu)
    stds.append(std)
    dfsub['nN'+str(i)] = (dfsub['N'+str(i)]-mu)/std #Normalizing.

for i in range(0,8):
    dfsub['nN'+str(i)] = (dfsub['N'+str(i)]-mus[i])/stds[i] #Normalizing.

dfsub.insert(dfsub.shape[1],'w_rad',dfsub['weight']*dfsub['radiance']) #Weighted radiance.

#Processing the gdp data.
dgsub = dfsub.groupby(['state','county']).agg('sum') #This is the gdp dataframe. I should rename this.
gdpsub = pd.read_csv(gdpsub_path, encoding='cp1252')
gdpsub['GDP20'] = gdpsub['GDP20'].replace(',','',regex=True).astype('float64')
gdpsub['GDP20adj'] = (2.14332E+13)*gdpsub['GDP20']/19032672000
dgsub = dgsub.join(gdpsub.set_index(['state','county']),how='inner')

m = LinearRegression().fit(np.log10(dgsub['w_rad']).values.reshape(-1,1),np.log10(dgsub['GDP20']).values)
m.score(np.log10(dgsub['w_rad']).values.reshape(-1,1),np.log10(dgsub['GDP20']).values)

def add_const(xdata,a,b):
    return np.log10(a*np.power(10,xdata)+b)

def get_subnational_R2(s,t,dfsub=dfsub,df=df):
    model = get_model(s,t,df)
    regionalize(dfsub,s,t)
    cols=[]
    for i in range(0,m):
        col = 'r'+str(i)+'_dash'
        dfsub[col] = dfsub['radiance']*dfsub['weight']*(dfsub['cluster_dash']==i)
        cols.append(col)
    dgsub = dfsub[['state','county']+cols].groupby(['state','county']).agg('sum')
    dgsub = gdpsub.join(dgsub,on=['state','county'],how='inner')
    X = dgsub[cols]
    X = np.log10(X+1).values
    Y = np.log10(dgsub['GDP20']).values
    Ydash = model.predict(X)#.reshape(-1,1)
    params,_ = curve_fit(add_const,Ydash,Y)
    a, b = params
    Ydash = add_const(Ydash,a,b)
    plt.scatter(Ydash,Y)
    return r2_score(Ydash,Y)

get_subnational_R2(optimal_s,optimal_t)

np.random.seed(9897)
    
def get_model_score(dfsub, s, t=None):
    m = regionalize(dfsub,s,t)
    cols=[]
    for i in range(0,m):
        col = 'r'+str(i)+'_dash'
        dfsub[col] = dfsub['radiance']*dfsub['weight']*(dfsub['cluster_dash']==i)
        cols.append(col)
    dgsub = dfsub[['state','county']+cols].groupby(['state','county']).agg('sum')
    dgsub = gdpsub.join(dgsub,on=['state','county'],how='inner')
    X = dgsub[cols]
    X = np.log10(X+1).values
    Y = np.log10(dgsub['GDP20']).values
    return get_model_CVMSE(X,Y)

def get_model_CVMSE(X,Y, k_folds=5):
    ordering = np.tile(range(0,k_folds),int(1 + len(Y)/k_folds))[0:len(Y)]
    np.random.shuffle(ordering)
    score = 0
    for i in range(0,k_folds):
        regress2 = LinearRegression().fit(X[~(ordering==i)],Y[~(ordering==i)])
        score += mean_squared_error(regress2.predict(X[ordering==i]),Y[ordering==i]) / k_folds
    return score

### Half the min_region_size because half the data.
#results = model_search_fast(dftsub,60000,sigma=1,epsilon=0.05,t=[0.1,0.1],s=[1,1,1,1,1],decay=True,min_region_size=126,beta=0.9997697679981565)

def get_model_R2(dfsub, s, t, dfvsub=None, model=None, graph=False):
    m=regionalize(dfsub,s,t)
    cols=[]
    for i in range(0,m):
        col = 'r'+str(i)+'_dash'
        dfsub[col] = dfsub['radiance']*dfsub['weight']*(dfsub['cluster_dash']==i)
        cols.append(col)
    dgsub = dfsub[['state','county']+cols].groupby(['state','county']).agg('sum')
    dgsub = gdpsub.join(dgsub,on=['state','county'],how='inner')
    X = dgsub[cols]
    X = np.log10(X+1).values
    Y = np.log10(dgsub['GDP20']).values
    print("Entering.")
    if model is not None:
        print("Model is not None.")
        if graph:
            plt.scatter(model.predict(X),Y)
        return model.score(X,Y)
    print(len(X))
    model = LinearRegression().fit(X,Y)
    if dfvsub is not None: #After fitting, the calculation needs to go through the above steps again for the validation set..
        print("dfv is not None.")
        return get_model_R2(dfvsub,s,t,model=model,graph=graph)
    else:
        print("dfv is None.")
        if graph:
            plt.scatter(model.predict(X),Y)
        return model.score(X,Y)
    
def model_search_fast(df, n_samp = 100, seed=4550, epsilon=0.8, sigma=0.05, s=None, t=[1,2], decay=False, beta=0.999, min_region_size=256):
    results = ([],[],[])
    np.random.seed(seed)
    rand = 0
    if s is not None:
        best_s = s
    else:
        best_s = np.random.normal(0,sigma*5,size=len(s))
    best_t = t
    minner = get_model_score(df,best_s,best_t)
    for i in tqdm(range(0,n_samp)):
        if rand < epsilon:
            s = np.random.normal(0,0.5,size=len(s))
            t = np.random.normal(1,0.5,size=len(t))
        else:
            s = best_s + np.random.normal(0,sigma,size=len(s))
            t = best_t + np.random.normal(0,sigma,size=len(t))
        LL = get_model_score(df,s,t)
        results[0].append(s)
        results[1].append(t)
        results[2].append(LL)
        if LL < minner:
            if df.groupby('cluster_dash').agg('count')['im_name'].min() >= min_region_size:
                print(t)
                print(s)
                print(LL)
                minner = LL
                best_s = s
                best_t = t
        if decay:
            sigma *= beta
    return best_s, best_t, minner


def loop_results(dfsub,init_s,init_t=[],max_regions=5,iters=1000,stopping_value=1e-2,min_region_size=256): #Get log-likelihood for increasing number of regions.
    Ss = []
    Ts = []
    LLs = []
    s = init_s
    t = init_t.copy()
    for i in range(len(init_t)+2,max_regions+1):
        beta = np.power(stopping_value,1/iters)
        results = model_search_fast(dfsub,iters,sigma=1,epsilon=0,t=t,s=s,decay=True,beta=beta,min_region_size=min_region_size)
        s = results[0]
        t = results[1]
        Ss.append(s)
        Ts.append(t)
        LLs.append(results[2])
        t = np.append(t,0.01)
    return Ss, Ts, LLs

def compute_AICs(dfsub,init_s=[0.5,0.5,0.5],max_dim=7,max_regions=5,iters=1000,stopping_value=1e-2,min_region_size=256):
    Ss = []
    Ts = []
    LLs = []
    AICs = []
    train_n = len(train_counties)
    init_s = init_s.copy()
    for n_dim in range(len(init_s)+1,max_dim+1):
        init_s.append(0.5)
        a,b,c = loop_results(dfsub,init_s,init_t=[],max_regions=max_regions,iters=iters,stopping_value=stopping_value,min_region_size=min_region_size)
        Ss = Ss+a
        Ts = Ts+b
        LLs = LLs+c
        for i in range(0,len(c)):
            k = len(init_s)+2*len(b[i])+2+2
            k_dash = k+(k**2 + k)/(train_n-k+1)
            ll = np.log(c[i])*train_n
            aic = ll + 2*k_dash
            AICs.append(aic)
            print("n_dim = {}, splits = {}, AICc = {}".format(str(n_dim),str(len(b[i])+1),str(aic)))
    return Ss, Ts, AICs

n_seeds=40
aic_iters=500
d, p, AICs = [None,None,None]
vR2s = []
R2s = []
for seed in range(0,n_seeds):
    np.random.seed(seed)
    validation_counties = np.random.choice(dgsub.index, 100, replace=False)
    train_counties = dgsub[~dgsub.index.isin(validation_counties)].index
    train_counties = [s+c for s,c in train_counties]
    validation_counties = [s+c for s,c in validation_counties]
    dfsub['id'] = dfsub['state']+dfsub['county']
    dftsub = dfsub.loc[dfsub.id.isin(train_counties)].copy() #Training sample points,
    dfvsub = dfsub.loc[dfsub.id.isin(validation_counties)].copy() #Training sample points,
    ss,ts,aics = compute_AICs(dftsub,iters=aic_iters,init_s=[1,0,0,0,0,0],max_dim=8)
    ind = np.argmin(aics)
    s = ss[ind]
    t = ts[ind]
    s,t,_ = model_search_fast(dftsub,n_samp=aic_iters*10,sigma=1,epsilon=0.05,t=t,s=s,decay=True,beta=(1e-2)**(1/aic_iters/10),min_region_size=256)
    vR2s.append(get_model_R2(dftsub,s,t,dfvsub=dfvsub))
    R2s.append(get_model_R2(dfsub,s,t))
    if d is None:
        d = [len(s) for s in ss]
        p = [len(t)+2 for t in ts]
        AICs = [[aic] for aic in aics]
    else:
        for i in range(len(aics)):
            AICs[i].append(aics[i])
    print("Seed " + str(seed) + " complete :)")
    
ss,ts,aics = compute_AICs(dfsub,iters=aic_iters,init_s=[1,0,0,0,0,0],max_dim=8)
optimal_s, optimal_t, _ = model_search_fast(dfsub,n_samp=aic_iters*10,sigma=1,epsilon=0.05,t=ts[ind],s=ss[ind],decay=True,beta=(1e-2)**(1/aic_iters/10),min_region_size=256)

### This is the version of regionalize where we drop the requirement that hyperplanes must be parallel to one another.
def regionalize(dfsub,s,t=None):
    n_rows = s.shape[0]
    dfsub['cluster_dash'] = 0
    cols = ['nN'+str(i) for i in range(0,s.shape[1])]
    for i in range(0,n_rows):
        dfsub['proj'] = dfsub[cols].dot(s[i,:]) #Numpy is so elegant at times.
        dfsub['cluster_dash'] += (2**i)*(dfsub['proj'] > 1)
    return 2**n_rows

n_seeds=40
aic_iters=500
d, p, AICs = [None,None,None]
vR2s = []
R2s = []
for seed in range(0,n_seeds):
    np.random.seed(seed)
    validation_counties = np.random.choice(dgsub.index, 100, replace=False)
    train_counties = dgsub[~dgsub.index.isin(validation_counties)].index
    train_counties = [s+c for s,c in train_counties]
    validation_counties = [s+c for s,c in validation_counties]
    dfsub['id'] = dfsub['state']+dfsub['county']
    dftsub = dfsub.loc[dfsub.id.isin(train_counties)].copy() #Training sample points,
    dfvsub = dfsub.loc[dfsub.id.isin(validation_counties)].copy() #Training sample points,
    s = np.random.random((2,7))
    t=[]
    s,_ = model_search_fast(dftsub,n_samp=aic_iters*10,sigma=1,epsilon=0.05,t=t,s=s,decay=True,beta=(1e-2)**(1/aic_iters/10),min_region_size=256)
    vR2s.append(get_model_R2(dftsub,s,t,dfvsub=dfvsub))
    R2s.append(get_model_R2(dfsub,s,t))
    if d is None:
        d = [len(s) for s in ss]
        p = [len(t)+2 for t in ts]
        AICs = [[aic] for aic in aics]
    else:
        for i in range(len(aics)):
            AICs[i].append(aics[i])
    print("Seed " + str(seed) + " complete :)")

def get_model_CVMSE(X,Y, k_folds=5):
    ordering = np.tile(range(0,k_folds),int(1 + len(Y)/k_folds))[0:len(Y)]
    np.random.shuffle(ordering)
    score = 0
    for i in range(0,k_folds):
        regress2 = LinearRegression().fit(X[~(ordering==i)],Y[~(ordering==i)])
        score += mean_squared_error(regress2.predict(X[ordering==i]),Y[ordering==i]) / k_folds
    return score

def model_search_fast(df, n_samp = 100, seed=4550, epsilon=0.8, sigma=0.05, s=None, t=[1,2], decay=False, beta=0.999, min_region_size=256):
    results = ([],[])
    np.random.seed(seed)
    rand = 0
    if s is not None:
        best_s = s
    else:
        best_s = np.random.normal(0,sigma*5,size=n_dim)
    minner = get_model_score(df,best_s,t)
    for i in tqdm(range(0,n_samp)):
        if rand < epsilon:
            s = np.random.normal(0,0.5,size=s.shape)
        else:
            s = best_s + np.random.normal(0,sigma,size=s.shape)
        LL = get_model_score(df,s,t)
        results[0].append(s)
        results[1].append(LL)
        if LL < minner:
            if df.groupby('cluster_dash').agg('count')['im_name'].min() >= min_region_size:
                print(s)
                print(LL)
                minner = LL
                best_s = s
        if decay:
            sigma *= beta
    return best_s, minner#results

### BOOTSTRAP VARIANCE ###
#This estimates the sampling error. It's just the variance with no bias because the estimator is unbiased.
def get_model(s, t, dfsub=dfsub):
    m = regionalize(dfsub,s,t)
    cols=[]
    for i in range(0,m):
        col = 'r'+str(i)+'_dash'
        dfsub[col] = dfsub['radiance']*dfsub['weight']*(dfsub['cluster_dash']==i)
        cols.append(col)
    dgsub = dfsub[['state','county']+cols].groupby(['state','county']).agg('sum')
    dgsub = gdpsub.join(dgsub,on=['state','county'],how='inner')
    X = dgsub[cols]
    X = np.log10(X+1).values
    Y = np.log10(dgsub['GDP20']).values
    model = LinearRegression().fit(X,Y)
    return model

np.random.seed(1849)
n_samp = 250
cnts = train_counties+validation_counties
model = get_model(s=optimal_s,t=optimal_t)
cols = ['r'+str(i)+'_dash' for i in range(0,2)]
replications = 50
county_ests = np.empty(len(cnts))
for j in range(0,len(cnts)):
    county_id = cnts[j]
    ests = np.empty(replications)
    for i in tqdm(range(0,replications)):
        county_dfsub = dfsub.loc[dfsub.id==county_id,['state','county']+cols]
        samp = county_dfsub.sample(n_samp,replace=True) #Selecting with equal probabiltites from the sample emulates the PPS sampling scheme as the sample is already PPS representative.
        X = samp.groupby(['state','county']).agg('sum')
        X = np.log10(X+1)
        ests[i] = model.predict(X)[0]
    county_ests[j] = np.sqrt(ests.var())/ests.mean()
    print(county_id+" complete.")
    print(county_ests[j]*100)
    
def make_scatters(s,t): #Produces Fig. 4 from the paper.
    get_model(s=s,t=t,dfsub=dfvsub)
    test_model = get_model(s=s,t=t,dfsub=dftsub)
    
    
    cols = ['r'+str(i)+'_dash' for i in range(0,len(t)+2)]
    
    final_model = get_model(s=s,t=t,dfsub=dfsub)
    
    #Calculates the predictions for the whole study area using the final model.
    dgsub = dfsub[['state','county']+cols].groupby(['state','county']).agg('sum')
    dgsub = gdpsub.join(dgsub,on=['state','county'],how='inner')
    X = dgsub[cols]
    X = np.log10(X+1).values
    Y = np.log10(dgsub['GDP20']).values
    
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
    dgsub = dfvsub[['state','county']+cols].groupby(['state','county']).agg('sum')
    dgsub = gdpsub.join(dgsub,on=['state','county'],how='inner')
    X = dgsub[cols]
    X = np.log10(X+1).values
    Y = np.log10(dgsub['GDP20']).values
    
    
    ax[0].scatter(test_model.predict(X),Y,s=5,alpha=0.7,edgecolors='None')
    ax[0].annotate("R^2 = {:.3f}".format(np.round(test_model.score(X,Y),3)), (8,13), fontsize=8)

    ax[0].set_xlim(lims)
    #ax[0].set_title("Predicted GDP vs actual GDP on validation set")
    ax[0].set(ylabel="log10 GDP",xlabel="log10 Predicted GDP")
    ax[0].plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=1)
    ax[0].set_aspect('equal')
    ax[0].set_ylim(lims)
    
    plt.savefig("scatters_hyperplane_sub.eps",format='eps')