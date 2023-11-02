import random
import pickle
import numpy as np
import scipy as sp
import pandas as pd
from sklearn import ensemble, linear_model



def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())

dens_levels = [2600,2700,2750,2770,2780,2790,2800,2805,2810,2820,2825,2830,2832]
df_empty = pd.DataFrame()
lst = [df_empty for x in range(len(dens_levels))]

for i in range(len(dens_levels)):
    lst[i] = pd.read_csv('../lc929/Data/Time_Correction/csv/global_halfdeg_3000yrs_csv/tracer_reconstr_gn'+str(dens_levels[i])+'.csv', sep ="\t")

df = pd.concat(lst, ignore_index=True, sort=False)
# # finding all data in a year
# # later using data from years before and after to teach model
# data_year = df.loc[(df['Year'] >= 1990) & (df['Year'] <= 2000)]
# # print(data_1989)


relevant = df[["Year","Longitude", "Latitude", "Gamman", "CFC11_rec", "CFC12_rec", "SF6_rec"]]
cleanrelevant = relevant.dropna(subset = ["CFC11_rec", "CFC12_rec", "SF6_rec"], inplace=False)
print(len(cleanrelevant))

# returns array of pearson, array of rmse tested on a random subset of testfrac samples
def corrtest(cleanrelevant, testfrac):
    # pick random rows to be testing data
    n = int(np.floor(len(cleanrelevant)*testfrac))
    indices = np.array(cleanrelevant.index).tolist()
    randomnumbers = random.sample(indices, n)
    testing = cleanrelevant.loc[randomnumbers]


    # training = cleanrelevant.drop([randomnumber], axis=0)
    training = cleanrelevant.drop(randomnumbers)

    clf = ensemble.RandomForestRegressor(n_jobs=-1)
    clf.fit(np.array(training[["Year","Longitude", "Latitude", "Gamman"]]), np.array(training[["CFC11_rec", "CFC12_rec", "SF6_rec"]]))
    # print("\"true\":", testing)
    predictions = np.array(clf.predict(np.array(testing[["Year","Longitude", "Latitude", "Gamman"]])))
    # print("predicted:", predictions)


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

pearsonsum, rmsesum = np.array([0.0,0,0]), np.array([0.0,0,0])
tries = 1
for i in range(tries):
    a,b = corrtest(cleanrelevant, 0.1)
    pearsonsum += a
    rmsesum +=b
print(pearsonsum/tries)
print(rmsesum/tries)
