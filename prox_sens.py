import random
import numpy as np
import scipy as sp
import pandas as pd
pd.options.mode.chained_assignment = None

from sklearn import ensemble
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


# df dataframe, model with .predict() method, quantity string, gamman float, filename str, plot_title str
def truepredplot(df, model, quantity, gamman, longrange, latrange, filename, plot_title):
    relevant = df[["Year","Longitude", "Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar", quantity]].dropna()
    i = ["CFC11_rec", "CFC12_rec", "SF6_rec"].index(quantity)
    blockrel = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])&(relevant["Gamman"]==gamman)]
    if len(blockrel) < 1:
        filename = directory + "/" + "insufficient.txt"
        with open(filename, 'w') as f:
            f.write('No data points for this block and density.')
    else:
        years = df.Year.unique()
        for y in years:
            relrows = blockrel[(blockrel["Year"]==y)]
            data = relrows[["Longitude", "Latitude", quantity]]
            prediction = np.array(model.predict(np.array(relrows[["Year","Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar"]]))[:,i])
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


            title = fig.suptitle(plot_title)
            if len(data) > 1:
                text = fig.text(0.50, 0.02, 'RMSE: '+ str(rmse(np.array(data[quantity]), np.array(data['predicted']))) + '\n Pearson: ' + str(sp.stats.pearsonr(np.array(data[quantity]), np.array(data['predicted']))[0]), horizontalalignment='center', wrap=False )

            plt.savefig(filename)
            plt.close()


def temporal(df, year, longrange, latrange, gn, dir, quantity, range):
    yearmax = year+range
    yearmin = year-range
    relevant = df[["Year","Longitude", "Long_x", "Long_y", "Latitude", "Gamman", "CFC11_rec", "CFC12_rec", "SF6_rec", "T_bar", "S_bar"]].dropna()
    testing = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])&(relevant["Year"]<=yearmax)&(relevant["Year"]>=yearmin)]
    training = relevant[~((relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])&(relevant["Year"]<=yearmax)&(relevant["Year"]>=yearmin))]
    clf = ensemble.RandomForestRegressor(min_samples_leaf=10,n_jobs=-1)
    clf.fit(np.array(training[["Year","Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar"]]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))
    truepredplot(relevant, clf, quantity, gn, longrange, latrange, dir+'/'+quantity+'_temporal', 'Year_'+ str(year)+' excluding '+str(longrange)+','+str(latrange)+', between '+str(yearmin)+' and '+str(yearmax))

def density(df, year, longrange, latrange, gn, dir, quantity, range):
    relevant = df[["Year","Longitude", "Long_x", "Long_y", "Latitude", "Gamman", "CFC11_rec", "CFC12_rec", "SF6_rec", "T_bar", "S_bar"]].dropna()
    gns = np.sort(np.array(relevant.Gamman.unique()))
    i = gns.tolist().index(gn)
    if i<range:
        imax = i+range
        imin = 0
    elif i+range>len(gns.tolist())-1:
        imin = i-range
        imax = len(gns.tolist())-1
    else:
        imax = i+range
        imin = i-range
    testing = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])&(relevant["Gamman"]<=gns[imax])&(relevant["Gamman"]>=gns[imin])]
    training = relevant[~((relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])&(relevant["Gamman"]<=gns[imax])&(relevant["Gamman"]>=gns[imin]))]
    clf = ensemble.RandomForestRegressor(min_samples_leaf=10,n_jobs=-1)
    clf.fit(np.array(training[["Year","Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar"]]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))
    truepredplot(relevant, clf, quantity, gn, longrange, latrange, dir+'/'+quantity+'_density', 'Year_'+ str(year)+' excluding '+str(longrange)+','+str(latrange)+', Gamman between '+str(gns[imin])+' and '+str(gns[imax]))

df = pd.read_csv('tracer_reconstr_ALL_TSavg_Longcircle.csv', sep ='\t')
print('read complete')

year=2010
longrange = [40,60]
latrange = [-30,-10]
gn = 27
dir = 'prox_sens_tests/three_levels_pm_gn2700_2010'
quantity = 'CFC11_rec'
range = 3

if not os.path.exists(dir):
    os.makedirs(dir)

temporal(df, year, longrange, latrange, gn, dir, quantity, range)
print('temporal complete')
density(df, year, longrange, latrange, gn, dir, quantity, range)
