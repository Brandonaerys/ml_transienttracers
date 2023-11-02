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

def convert_longitude(value):
    if value > 180:
        return value - 360
    return value


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())

# returns model
def rf_regtree(df, year, longrange, latrange, tree_number, min_samples_leaf, singleyear=False, with_O=True, projectcircle=True, fliplong=True):
    if singleyear:
        df = df[(df['Year']==year)]
    if fliplong:
        df['Longitude'] = df['Longitude'].apply(convert_longitude)
    if projectcircle:
        if with_O:
            predictors = ["Year", "Long_x", "Long_y", "Latitude", "Gamman","T_bar", "S_bar", 'O_bar']
            relevant = df[["Year","Longitude", "Long_x", "Long_y", "Latitude", "Gamman", "CFC11_rec", "CFC12_rec", "SF6_rec", "T_bar", "S_bar", 'O_bar']].dropna()
            testing = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])]
            training = relevant[~((relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1]))]
            clf = ensemble.RandomForestRegressor(n_estimators=tree_number, min_samples_leaf=min_samples_leaf,n_jobs=-1)
            clf.fit(np.array(training[predictors]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))
        else:
            predictors = ["Year", "Long_x", "Long_y", "Latitude", "Gamman","T_bar", "S_bar"]
            relevant = df[["Year","Longitude", "Long_x", "Long_y", "Latitude", "Gamman", "CFC11_rec", "CFC12_rec", "SF6_rec", "T_bar", "S_bar"]].dropna()
            testing = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])]
            training = relevant[~((relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1]))]
            clf = ensemble.RandomForestRegressor(n_estimators=tree_number, min_samples_leaf=min_samples_leaf,n_jobs=-1)
            clf.fit(np.array(training[predictors]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))
    else:

        if with_O:
            predictors = ["Year", "Longitude", "Latitude", "Gamman","T_bar", "S_bar", 'O_bar']
            relevant = df[["Year","Longitude", "Latitude", "Gamman", "CFC11_rec", "CFC12_rec", "SF6_rec", "T_bar", "S_bar", 'O_bar']].dropna()
            testing = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])]
            training = relevant[~((relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1]))]
            clf = ensemble.RandomForestRegressor(n_estimators=tree_number, min_samples_leaf=min_samples_leaf,n_jobs=-1)
            clf.fit(np.array(training[predictors]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))
        else:
            predictors = ["Year","Longitude", "Latitude", "Gamman", "T_bar", "S_bar"]
            relevant = df[["Year","Longitude", "Latitude", "Gamman", "CFC11_rec", "CFC12_rec", "SF6_rec", "T_bar", "S_bar"]].dropna()
            testing = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])]
            training = relevant[~((relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1]))]
            clf = ensemble.RandomForestRegressor(n_estimators=tree_number, min_samples_leaf=min_samples_leaf,n_jobs=-1)
            clf.fit(np.array(training[predictors]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))


    return clf, predictors





def truepredplot(df, model, predictors, quantity, gamman, year, longrange, latrange, directory, filename, titletext="title", singleyear=False, fliplong=True):
    if singleyear:
        df = df[(df['Year']==year)]
    if fliplong:
        df['Longitude'] = df['Longitude'].apply(convert_longitude)
    if not os.path.exists(directory):
        os.makedirs(directory)
    keepcols = predictors.copy()
    keepcols.append(quantity)
    if not 'Longitude' in keepcols:
        keepcols.append('Longitude')
    relevant = df[keepcols].dropna()
    i = ["CFC11_rec", "CFC12_rec", "SF6_rec"].index(quantity)
    blockrel = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])&(relevant["Gamman"]==gamman)]
    relrows = blockrel[(blockrel["Year"]==year)]
    data = relrows[["Longitude", "Latitude", quantity]]
    prediction = np.array(model.predict(np.array(relrows[predictors]))[:,i])
    data['predicted'] = prediction
    data['true-pred_percent'] = (np.array(data[[quantity]].copy().values - data[['predicted']].copy().values)/np.array(data[[quantity]].copy().values)) * 100

    min = data[[quantity, 'predicted']].to_numpy().min()
    max = data[[quantity, 'predicted']].to_numpy().max()


    parallels = np.arange(latrange[0],latrange[1],5.)
    meridians = np.arange(longrange[0],longrange[1],5.)


    fig, axs = plt.subplots(2, 2, figsize=(15,10))


    map1 = Basemap(llcrnrlon=longrange[0], llcrnrlat=latrange[0], urcrnrlon=longrange[1], urcrnrlat=latrange[1],  projection='merc', ax = axs[0,0])
    map1.drawcoastlines(zorder=1)
    map1.drawparallels(parallels,labels=[True,False,True,False], zorder=2)
    map1.drawmeridians(meridians,labels=[False,True,False,True], zorder=3)
    x1,y1 = map1(data['Longitude'],data['Latitude'])
    scatter1 = axs[0,0].scatter(x1,y1,c=data[quantity], cmap='inferno', vmin = min, vmax = max, zorder=4)
    axs[0,0].set_title('true values')
    cbar1 = fig.colorbar(scatter1, ax = axs[0,0])
    axs[0,0].set_axisbelow(True)


    map2 = Basemap(llcrnrlon=longrange[0], llcrnrlat=latrange[0], urcrnrlon=longrange[1], urcrnrlat=latrange[1],  projection='merc', ax = axs[0,1])
    map2.drawcoastlines(zorder=1)
    map2.drawparallels(parallels,labels=[True,False,True,False], zorder=2)
    map2.drawmeridians(meridians,labels=[False,True,False,True], zorder=3)
    x2,y2 = map2(data['Longitude'],data['Latitude'])
    scatter2 = axs[0, 1].scatter(x2,y2,c=data['predicted'], cmap='inferno', vmin = min, vmax = max, zorder=4)
    axs[0,1].set_title('predicted values')
    cbar2 = fig.colorbar(scatter2, ax = axs[0,1])
    axs[0,1].set_axisbelow(True)


    map3 = Basemap(llcrnrlon=longrange[0], llcrnrlat=latrange[0], urcrnrlon=longrange[1], urcrnrlat=latrange[1],  projection='merc', ax = axs[1,0])
    map3.drawcoastlines(zorder=1)
    map3.drawparallels(parallels,labels=[True,False,True,False], zorder=2)
    map3.drawmeridians(meridians,labels=[False,True,False,True], zorder=3)
    x3,y3 = map3(data['Longitude'],data['Latitude'])
    scatter3 = axs[1,0].scatter(x3,y3,c=data['true-pred_percent'], cmap='RdYlBu', vmin = -200, vmax = 200, zorder=4)
    axs[1,0].set_title('percentage diff (true-pred)')
    cbar3 = fig.colorbar(scatter3, ax = axs[1,0])
    axs[1,0].set_axisbelow(True)



    axs[1,1].scatter(data[quantity], data['predicted'])
    line_x = np.linspace(0, max, 100)
    line_y = line_x
    axs[1,1].plot(line_x, line_y, color='green')
    axs[1,1].set_title('true vs predicted')
    axs[1,1].grid()
    axs[1,1].set_axisbelow(True)


    title = fig.suptitle(titletext)
    if len(data) > 1:
        text = fig.text(0.50, 0.02, 'RMSE: '+ str(rmse(np.array(data[quantity]), np.array(data['predicted']))) + '\n Pearson: ' + str(sp.stats.pearsonr(np.array(data[quantity]), np.array(data['predicted']))[0]), horizontalalignment='center', wrap=False )

    plt.savefig(directory+'/'+filename)
    plt.close()








def alltests(df, longrange, latrange, gns, year, quantity, singleyear=False, fliplong=True):
    # vary number of trees
    for gamman in gns:
        directory = 'seg_tests/long'+str(longrange[0])+str(longrange[1])+'_'+'lat'+str(latrange[0])+str(latrange[1])+'_'+'gn'+str(int(gamman*100))+'_'+str(year)
        if singleyear:
            directory += '/singleyear'

        model, predictors = rf_regtree(df, year, longrange, latrange, tree_number=1, min_samples_leaf=10, singleyear=singleyear, with_O=False, projectcircle=True, fliplong=True)
        truepredplot(df, model, predictors, quantity, gamman, year, longrange, latrange, directory, 'trees_1', titletext="tree_number=1, min_samples_leaf=10, with_O=False, projectcircle=True", singleyear=True, fliplong=True)
        print('one tree complete')


        model, predictors = rf_regtree(df, year, longrange, latrange, tree_number=100, min_samples_leaf=10, singleyear=singleyear, with_O=False, projectcircle=True, fliplong=True)
        truepredplot(df, model, predictors, quantity, gamman, year, longrange, latrange, directory, 'baseline', titletext="tree_number=100, min_samples_leaf=10, with_O=False, projectcircle=True", singleyear=True, fliplong=True)

        model, predictors = rf_regtree(df, year, longrange, latrange, tree_number=1000, min_samples_leaf=10, singleyear=singleyear, with_O=False, projectcircle=True, fliplong=True)
        truepredplot(df, model, predictors, quantity, gamman, year, longrange, latrange, directory, 'trees_1000', titletext="tree_number=1000, min_samples_leaf=10, with_O=False, projectcircle=True", singleyear=True, fliplong=True)

        # vary min_samples_leaf
        model, predictors = rf_regtree(df, year, longrange, latrange, tree_number=100, min_samples_leaf=1, singleyear=singleyear, with_O=False, projectcircle=True, fliplong=True)
        truepredplot(df, model, predictors, quantity, gamman, year, longrange, latrange, directory, 'minleafsize_1', titletext="tree_number=100, min_samples_leaf=1, with_O=False, projectcircle=True", singleyear=True, fliplong=True)

        model, predictors = rf_regtree(df, year, longrange, latrange, tree_number=100, min_samples_leaf=50, singleyear=singleyear, with_O=False, projectcircle=True, fliplong=True)
        truepredplot(df, model, predictors, quantity, gamman, year, longrange, latrange, directory, 'minleafsize_50', titletext="tree_number=100, min_samples_leaf=50, with_O=False, projectcircle=True", singleyear=True, fliplong=True)

        model, predictors = rf_regtree(df, year, longrange, latrange, tree_number=100, min_samples_leaf=100, singleyear=singleyear, with_O=False, projectcircle=True, fliplong=True)
        truepredplot(df, model, predictors, quantity, gamman, year, longrange, latrange, directory, 'minleafsize_100', titletext="tree_number=100, min_samples_leaf=100, with_O=False, projectcircle=True", singleyear=True, fliplong=True)

        # vary with_O
        model, predictors = rf_regtree(df, year, longrange, latrange, tree_number=100, min_samples_leaf=10, singleyear=singleyear, with_O=True, projectcircle=True, fliplong=True)
        truepredplot(df, model, predictors, quantity, gamman, year, longrange, latrange, directory, 'withO', titletext="tree_number=100, min_samples_leaf=10, with_O=True, projectcircle=True", singleyear=True, fliplong=True)

        # vary projectcircle
        model, predictors = rf_regtree(df, year, longrange, latrange, tree_number=100, min_samples_leaf=10, singleyear=singleyear, with_O=False, projectcircle=False, fliplong=True)
        truepredplot(df, model, predictors, quantity, gamman, year, longrange, latrange, directory, filename='without_circle_projection', titletext="tree_number=100, min_samples_leaf=10, with_O=False, projectcircle=False", singleyear=True, fliplong=True)



df = pd.read_csv('tracer_reconstr_ALL_TSOavg_Longcircle.csv', sep ='\t')
print('read complete')
# # df2 has atlantic only
# df2 = df[(df["Longitude"]<20)|(df["Longitude"]>260)]

# rf_regtree(df, year, longrange, latrange, tree_number, min_samples_leaf, singleyear=False, with_O=True, projectcircle=True, fliplong=True)
# truepredplot(df, model, predictors, quantity, gamman, year, longrange, latrange, directory, filename, titletext="title", singleyear=False, fliplong=True)

longrange = [-80,-60]
latrange = [30,50]
gamman = [27.9]
year = 2013
quantity = 'CFC11_rec'

alltests(df, longrange, latrange, gns, year, quantity, singleyear=False, fliplong=True)
