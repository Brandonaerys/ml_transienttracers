import numpy as np
# import scipy as sp
import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
# import pickle




# data2750 = pd.read_csv('tracer_reconstr_gn2750.csv', sep ="\t")
# data2800 = pd.read_csv('tracer_reconstr_gn2800.csv', sep ="\t")
# data2832 = pd.read_csv('tracer_reconstr_gn2832.csv', sep ="\t")
# df = pd.concat([data2750, data2800, data2832], ignore_index=True, sort=False)
#
#
# relevant = df[["Year","Longitude", "Latitude", "Gamman", "CFC11_rec", "CFC12_rec", "SF6_rec"]]
# cleanrelevant = relevant.dropna(subset = ["CFC11_rec", "CFC12_rec", "SF6_rec"], inplace=False)
# testing = cleanrelevant[(cleanrelevant["Longitude"]>180)&(cleanrelevant["Longitude"]<200)&(cleanrelevant["Latitude"]>0)&(cleanrelevant["Latitude"]>20)]
# training = cleanrelevant[~((cleanrelevant["Longitude"]>180)&(cleanrelevant["Longitude"]<200)&(cleanrelevant["Latitude"]>0)&(cleanrelevant["Latitude"]>20))]
# print(training)
# print(len(cleanrelevant))
# print(len(training) + len(testing))

# df = pd.DataFrame(np.random.randint(0,100,size=(100, 2)), columns=list('AB'))
# df.to_csv('testing.csv', index=False, sep = '\t')
#
# df = pd.read_csv('testing.csv', sep ="\t")
# print(df[~((df["A"]<50)&(df["B"]<50))])

# T = np.empty((721, 361, ))
# T[:] = np.nan
# print(T)

# longrange = np.linspace(0,360, num=721)
# print(longrange)
#
# print(int(35)-160.1)

# print(np.radians(40))
#
# df = pd.read_csv('tracer_reconstr_gn2832_TSavg_Longcircle.csv', sep ='\t')
# print(df[['Year', 'Long_y', 'Longitude']].head())

# # a = np.array([0,1,2])
# # pickle.dump(a, open("testing.txt", 'wb'))
#
# b = pickle.load(open("testing.txt", "rb"))
# print(b*2)

# longblocks = 18
# longrange = np.linspace(0,360,num=longblocks+1,endpoint=True)
# print(longrange)
# print(int(36)==36.0)

# a = ["a string"]
# for i in a:
#     print(i)

# print(np.mod([-11.9, -12.1], 36))




#
# fig, axs = plt.subplots(2,2,figsize=(15,10))
#
# longrange = [0,20]
# latrange = [-50,-30]
#
# relevant = pd.read_csv('tracer_reconstr_gn2750.csv', sep ='\t')
# data = relevant[(relevant["Longitude"]>longrange[0])&(relevant["Longitude"]<longrange[1])&(relevant["Latitude"]>latrange[0])&(relevant["Latitude"]<latrange[1])]
#
# parallels = np.arange(latrange[0],latrange[1],5.)
# meridians = np.arange(longrange[0],longrange[1],5.)
# labels = [left,right,top,bottom]


#
# map1 = Basemap(llcrnrlon=longrange[0], llcrnrlat=latrange[0], urcrnrlon=longrange[1], urcrnrlat=latrange[1],  projection='merc', ax = axs[0,0])
# map1.drawcoastlines(zorder=1)
# map1.drawparallels(parallels,labels=[True,False,True,False], zorder=2)
# map1.drawmeridians(meridians,labels=[False,True,False,True], zorder=3)
#
#
# x1,y1 = map1(data['Longitude'],data['Latitude'])
# scatter1 = axs[0,0].scatter(x1,y1,c=data['CFC11_rec'], cmap='inferno', vmin = 0, vmax = 5, zorder=4)
# axs[0,0].set_title('merc')
# cbar1 = fig.colorbar(scatter1, ax = axs[0,0])
# # axs[0,0].grid(zorder=2)
# axs[0,0].set_axisbelow(True)
#
#
#
#
# map2 = Basemap(llcrnrlon=longrange[0], llcrnrlat=latrange[0], urcrnrlon=longrange[1], urcrnrlat=latrange[1],  projection='merc', ax = axs[0,1])
# map2.drawcoastlines(zorder=1)
# map2.drawparallels(parallels,labels=[True,False,True,False], zorder=2)
# map2.drawmeridians(meridians,labels=[False,True,False,True], zorder=3)
#
# x2,y2 = map2(data['Longitude'],data['Latitude'])
# scatter1 = axs[0,1].scatter(x2,y2,c=data['CFC11_rec'], cmap='inferno', zorder=4)
# axs[0,1].set_title('merc')
# cbar1 = fig.colorbar(scatter1, ax = axs[0,1])
# # axs[0,1].grid(zorder=2)
# axs[0,1].set_axisbelow(True)
#
#
#
#
#
# map3 = Basemap(llcrnrlon=longrange[0], llcrnrlat=latrange[0], urcrnrlon=longrange[1], urcrnrlat=latrange[1],  projection='merc', ax = axs[1,0])
# map3.drawcoastlines(zorder=1)
# map3.drawparallels(parallels,labels=[True,False,True,False], zorder=2)
# map3.drawmeridians(meridians,labels=[False,True,False,True], zorder=3)
#
# x3,y3 = map3(data['Longitude'],data['Latitude'])
# scatter1 = axs[1,0].scatter(x3,y3,c=data['CFC11_rec'], cmap='inferno', zorder=4)
# axs[1,0].set_title('merc')
# cbar1 = fig.colorbar(scatter1, ax = axs[1,0])
# # axs[1,0].grid(zorder=2)
# axs[1,0].set_axisbelow(True)
#
#
#
#
#
# map4 = Basemap(llcrnrlon=longrange[0], llcrnrlat=latrange[0], urcrnrlon=longrange[1], urcrnrlat=latrange[1],  projection='merc', ax = axs[1,1])
# map4.drawcoastlines(zorder=1)
# map4.drawparallels(parallels,labels=[True,False,True,False], zorder=2)
# map4.drawmeridians(meridians,labels=[False,True,False,True], zorder=3)
#
# x,y = map4(data['Longitude'],data['Latitude'])
# scatter1 = axs[1,1].scatter(x,y,c=data['CFC11_rec'], cmap='inferno', zorder=4)
# axs[1,1].set_title('true values')
# cbar1 = fig.colorbar(scatter1, ax = axs[1,1])
# # axs[1,1].grid(zorder=2)
# axs[1,1].set_axisbelow(True)
#
# fig.text(0.5, 0.02,"testing \n testing")
#
# plt.show()

# a = np.array([28.00, 28.32, 27.50, 27.1])
# print(np.sort(a).tolist().index(27.50))

# a = np.array([[1,np.nan,3,5], [2,3,4,5]])
# for i in range(2):
#     a = np.vstack((a,[1,2,3,4]))
#
# print(a)
# clean = a[~np.isnan(a).any(axis=1)]
# print(sp.stats.trim_mean(a[~np.isnan(a).any(axis=1)], 0))


# # Create some mock data
# t = np.arange(0, 10.0, 2)
# data1 = np.exp(t)
# data2 = np.sin(2 * np.pi * t)
#
# my_xticks = ['John','Arnold','Mavis','Matt', 'a']
#
#
#
# fig, ax1 = plt.subplots()
#
# ax1.set_xticks(t, my_xticks)
#
# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('exp', color=color)
# ax1.plot(t, data1, color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:blue'
# ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, data2, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

#
#
# def summer(a,b,c=10, d=10):
#     print(a,b,c,d)
#
# c = 5
# summer(b = 1,a=2, c=c,d=5)


# longrange = [280,300]
# latrange = [30,50]
# gamman = 27.9
# year = 2013
# directory = 'seg_tests/long'+str(longrange[0])+str(longrange[1])+'_'+'lat'+str(latrange[0])+str(latrange[1])+'_'+'gn'+str(int(gamman*100))+'_'+str(year)
# print(directory)


# # Sample DataFrame with 'Longitude' column ranging from 0 to 360
# data = {'Longitude': [10, 180, 270, 350]}
# df = pd.DataFrame(data)
#
# Function to convert longitude values from 0-360 to -180-180
# def convert_longitude(value):
#     if value > 180:
#         return value - 360
#     return value

# # Apply the conversion function to the 'Longitude' column
# df['Longitude'] = df['Longitude'].apply(convert_longitude)
#
# print(df)

# Create a 3x4 DataFrame with random values
data = np.random.rand(10, 4)
columns = ['A', 'B', 'C', 'D']

df = pd.DataFrame(data, columns=columns)

print(df)
df = df[(df['A']>0.5)]
print(df)
