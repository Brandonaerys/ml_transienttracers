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

def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


# Random Forest Model (predictors: Year,Longitude,Latitude,Gamman | output: CFC11_rec,CFC12_rec,SF6_rec)
# args: training data (dataframe), testing data (dataframe)
# returns: array of pearson, array of rmse
def corrtest(training, testing):

    clf = ensemble.RandomForestRegressor(min_samples_leaf=10,n_jobs=-1)

    # train with long
    # clf.fit(np.array(training[["Year","Longitude", "Latitude", "Gamman", "T_bar", "S_bar"]]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))
    # predictions = np.array(clf.predict(np.array(testing[["Year","Longitude", "Latitude", "Gamman", "T_bar", "S_bar"]])))

    # train with long projected to unit circle
    clf.fit(np.array(training[["Year","Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar"]]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))
    predictions = np.array(clf.predict(np.array(testing[["Year","Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar"]])))


    # pickle.dump(clf, open("rf_regtree_gn2830.txt", "wb"))
    # clf2 = pickle.load(open("rf_regtree_gn2830.txt", "rb"))

    # regression, rmse
    truevals = np.array(testing[["CFC11_rec", "CFC12_rec", "SF6_rec"]])

    cfc11cor = sp.stats.pearsonr(truevals[:,0], predictions[:,0])
    cfc12cor = sp.stats.pearsonr(truevals[:,1], predictions[:,1])
    sf6cor = sp.stats.pearsonr(truevals[:,2], predictions[:,2])

    # print("Pearson:", [cfc11cor[0], cfc12cor[0], sf6cor[0]])
    # print("rmse:", [rmse(truevals[:,0], predictions[:,0]), rmse(truevals[:,1], predictions[:,1]), rmse(truevals[:,2], predictions[:,2])])
    return np.array([cfc11cor[0], cfc12cor[0], sf6cor[0]]), np.array([rmse(truevals[:,0], predictions[:,0]), rmse(truevals[:,1], predictions[:,1]), rmse(truevals[:,2], predictions[:,2])])


def truepredplot(df, model, quantity, gamman, longrange, latrange, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    relevant = df[["Year","Longitude", "Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar", "O_bar", quantity]].dropna()
    i = ["CFC11_rec", "CFC12_rec", "SF6_rec"].index(quantity)
    blockrel = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])&(relevant["Gamman"]==gamman)]
    if len(blockrel) < 1:
        filename = directory + "/" + "insufficient.txt"
        with open(filename, 'w') as f:
            f.write('No data points for this block and density.')
    else:
        years = df.Year.unique()
        for y in years:
            filename = directory+'/'+str(y)+'_plot.png'
            relrows = blockrel[(blockrel["Year"]==y)]
            data = relrows[["Longitude", "Latitude", quantity]]
            prediction = np.array(model.predict(np.array(relrows[["Year","Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar", "O_bar"]]))[:,i])
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


            title = fig.suptitle(quantity + ' ' + str(y) + ' at ' + str(gamman) + ', predicted with TSO', fontsize=16)
            if len(data) > 1:
                text = fig.text(0.50, 0.02, 'RMSE: '+ str(rmse(np.array(data[quantity]), np.array(data['predicted']))) + '\n Pearson: ' + str(sp.stats.pearsonr(np.array(data[quantity]), np.array(data['predicted']))[0]), horizontalalignment='center', wrap=False )

            plt.savefig(filename)
            plt.close()




# dens_levels = [2600,2700,2750,2770,2780,2790,2800,2805,2810,2820,2825,2830,2832]
# df_empty = pd.DataFrame()
# lst = [df_empty for x in range(len(dens_levels))]

# for i in range(len(dens_levels)):
#     lst[i] = pd.read_csv('../lc929/Data/Time_Correction/csv/global_halfdeg_3000yrs_csv/tracer_reconstr_gn'+str(dens_levels[i])+'.csv', sep ="\t")

# df = pd.concat(lst, ignore_index=True, sort=False)

df = pd.read_csv('tracer_reconstr_ALL_TSOavg_Longcircle.csv', sep ='\t')
print("read complete")

relevant = df[["Year","Longitude", "Long_x", "Long_y", "Latitude", "Gamman", "CFC11_rec", "CFC12_rec", "SF6_rec", "T_bar", "S_bar", "O_bar"]].dropna()

# specify long and lat block to be removed
# testing = relevant[(relevant["Longitude"]>180)&(relevant["Longitude"]<190)&(relevant["Latitude"]>0)&(relevant["Latitude"]<10)&(relevant["Gamman"]==28)]
# training = relevant[~((relevant["Longitude"]>180)&(relevant["Longitude"]<190)&(relevant["Latitude"]>0)&(relevant["Latitude"]<10)&(relevant["Gamman"]==28))]

testing = relevant[(relevant["Longitude"]>0)&(relevant["Longitude"]<20)&(relevant["Latitude"]>-50)&(relevant["Latitude"]<-30)]
training = relevant[~((relevant["Longitude"]>0)&(relevant["Longitude"]<20)&(relevant["Latitude"]>-50)&(relevant["Latitude"]<-30))]
gns = relevant.Gamman.unique()

longs = [320,340]
lats=[10,30,50]

for i in range(len(longs)-1):
    for j in range(len(lats)-1):
        directory = 'regtree_plots/long'+str(int(longs[i]))+str(int(longs[i+1]))+'_lat'+str(int(lats[j]))+str(int(lats[j+1]))
        if not os.path.exists(directory):
            os.makedirs(directory)
        testing = relevant[(relevant["Longitude"]>longs[i])&(relevant["Longitude"]<longs[i+1])&(relevant["Latitude"]>lats[j])&(relevant["Latitude"]<lats[j+1])]
        training = relevant[~((relevant["Longitude"]>longs[i])&(relevant["Longitude"]<longs[i+1])&(relevant["Latitude"]>lats[j])&(relevant["Latitude"]<lats[j+1]))]
        if len(testing)>50:
            clf = ensemble.RandomForestRegressor(min_samples_leaf=30,n_jobs=-1)
            clf.fit(np.array(training[["Year","Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar", 'O_bar']]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))
            for g in gns:
                subdir = directory+'/'+str(int(g*100))+'_leaf30_withO'
                truepredplot(relevant, clf, 'CFC11_rec', g, [longs[i],longs[i+1]], [lats[j],lats[j+1]], subdir)

truepredplot(relevant, clf, "CFC11_rec", 27.0, [0,20], [-50, -30], directory)
