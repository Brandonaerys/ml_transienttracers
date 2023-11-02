import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


def loader(filename):
    return pickle.load(open(filename, 'rb'))

def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())

# df dataframe, model with .predict() method, quantity string, gamman float
def truepredplot(df, model, quantity, gamman, longrange, latrange):
    relevant = df[["Year","Longitude", "Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar", quantity]].dropna()
    i = ["CFC11_rec", "CFC12_rec", "SF6_rec"].index(quantity)
    blockrel = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])&(relevant["Gamman"]==gamman)]
    years = df.Year.unique()
    for y in years:
        filename = 'ABANDONED_blockwise_levelwise/long020_lat-70-50/'+str(y)+'_gn'+str(int(gamman*100))+'_plot.png'
        relrows = blockrel[(blockrel["Year"]==y)]
        data = relrows[["Longitude", "Latitude", quantity]]
        prediction = np.array(model.predict(np.array(relrows[["Year","Long_x", "Long_y", "Latitude", "Gamman", "T_bar", "S_bar"]]))[:,i])
        data['predicted'] = prediction
        data['true-pred_percent'] = (np.array(data[[quantity]].copy().values - data[['predicted']].copy().values)/np.array(data[[quantity]].copy().values)) * 100

        min = data[[quantity, 'predicted']].to_numpy().min()
        max = data[[quantity, 'predicted']].to_numpy().max()

        fig, axs = plt.subplots(2, 2, figsize=(15,10))
        scatter1 = axs[0,0].scatter(data['Longitude'],data['Latitude'],c=data[quantity], cmap='inferno', vmin = min, vmax = max)
        axs[0,0].set_title('true values')
        cbar1 = fig.colorbar(scatter1, ax = axs[0,0])


        scatter2 = axs[0, 1].scatter(data['Longitude'],data['Latitude'],c=data['predicted'], cmap='inferno', vmin = min, vmax = max)
        axs[0,1].set_title('predicted values')
        cbar2 = fig.colorbar(scatter2, ax = axs[0,1])

        scatter3 = axs[1,0].scatter(data['Longitude'],data['Latitude'],c=data['true-pred_percent'], cmap='seismic')
        axs[1,0].set_title('percentage diff (true-pred)')
        cbar3 = fig.colorbar(scatter3, ax = axs[1,0])

        axs[1,1].scatter(data[quantity], data['predicted'])
        line_x = np.linspace(0, max, 100)
        line_y = line_x
        axs[1,1].plot(line_x, line_y, color='green')
        axs[1,1].set_title('true vs predicted')

        title = fig.suptitle(quantity + ' ' + str(y) + ' at ' + str(gamman), fontsize=16)
        text = fig.text(0.50, 0.1, 'RMSE: '+ str(rmse(np.array(data[quantity]), np.array(data['predicted']))) + '\n Pearson: ' + str(sp.stats.pearsonr(np.array(data[quantity]), np.array(data['predicted']))[0]), horizontalalignment='center', wrap=True )

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(str(y)+' complete')


df = pd.read_csv('tracer_reconstr_ALL_TSavg_Longcircle.csv', sep ='\t')
print('read complete')
clf = loader('ABANDONED_blockwise_levelwise/long020_lat-70-50/rf_regtree_gn2700.txt')
truepredplot(df, clf, "CFC11_rec", 28, [0,20], [-50,90])
