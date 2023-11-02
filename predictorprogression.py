import random
import pickle
import numpy as np
import scipy as sp
import pandas as pd
pd.options.mode.chained_assignment = None

from sklearn import ensemble
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os



def loader(filename):
    return pickle.load(open(filename, 'rb'))

# a true, b observed
def nrmse(expected, obs):
    a,b = np.array(expected), np.array(obs)
    return np.sqrt(((a - b) ** 2).mean())/np.mean(b)

def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())



# df dataframe, gamman value, longrange list, latrange list,
def progstats(df, longrange, latrange, gamman, year, quantity):
    i = ["CFC11_rec", "CFC12_rec", "SF6_rec"].index(quantity)
    predictors = ['Year', 'Long_x', 'Long_y', 'Latitude', 'Gamman', 'T_bar', 'S_bar', 'O_bar']
    gns = df.Gamman.unique()
    relevant = df[["Year","Longitude", "Long_x", "Long_y", "Latitude", "Gamman", "CFC11_rec", "CFC12_rec", "SF6_rec", "T_bar", "S_bar", "O_bar"]].dropna()
    testing = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])&(relevant['Year']==year)&(relevant['Gamman']==gamman)]
    training = relevant[~((relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1]))]
    nrmses = np.array([np.nan for x in range(4)])
    pearsons = np.array([np.nan for x in range(4)])
    if len(testing)>1:
        for j in range(5, len(predictors)+1):
            used_predictors = predictors[0:j]
            clf = ensemble.RandomForestRegressor(min_samples_leaf=30,n_jobs=-1)
            clf.fit(np.array(training[used_predictors]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))
            predictions = np.array(clf.predict(np.array(testing[used_predictors]))[:,i])
            testing['predicted'] = predictions
            nrmses[j-5] = nrmse(np.array(testing[quantity]), np.array(testing['predicted']))
            pearsons[j-5] = sp.stats.pearsonr(np.array(testing[quantity]), np.array(testing['predicted']))[0]


    return nrmses, pearsons

def avgplot(df, longblocks, latblocks, year=2021, quantity = 'CFC11_rec'):
    longs = np.linspace(0,360,num=longblocks+1,endpoint=True)
    lats = np.linspace(-90,90,num=latblocks+1,endpoint=True)
    xticks = ['Year_Latlong_Density', 'with T', 'with S', 'with O']
    nrmse_arr = np.array([np.nan for x in range(4)])
    pearson_arr = np.array([np.nan for x in range(4)])
    for i in range(len(longs)-1):
        for j in range(len(lats)-1):
            longrange = [longs[i], longs[i+1]]
            latrange = [lats[j], lats[j+1]]
            for g in df.Gamman.unique():
                ntemp, ptemp = progstats(df, longrange, latrange, g, year, quantity)
                nrmse_arr = np.vstack((nrmse_arr,ntemp))
                pearson_arr = np.vstack((pearson_arr, ptemp))

    nrmse_avgs = sp.stats.trim_mean(nrmse_arr[~np.isnan(nrmse_arr).any(axis=1)], 0.1)
    pearson_avgs = sp.stats.trim_mean(pearson_arr[~np.isnan(pearson_arr).any(axis=1)], 0.1)

    filename = 'predictor_progression_stats.png'
    x = np.array([0,1,2,3])

    fig, ax1 = plt.subplots()

    ax1.set_xticks(x, xticks)

    color = 'tab:red'
    ax1.set_ylabel('Normalized rmse', color=color)
    ax1.plot(x, nrmse_avgs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Pearson', color=color)
    ax2.plot(x, pearson_avgs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    title = fig.suptitle('Change in NRMSE and Pearson with additional predictors, trimmed(0.1) mean, year: '+str(year), fontsize=16)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()








df = pd.read_csv('tracer_reconstr_ALL_TSOavg_Longcircle.csv', sep ='\t')
print('read complete')

avgplot(df, 18, 9)
