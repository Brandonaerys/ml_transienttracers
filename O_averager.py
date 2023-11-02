import numpy as np
import pandas as pd



df = pd.read_csv('tracer_reconstr_gn2750_TSavg_Longcircle.csv', sep ="\t")
# df = pd.read_csv('tracer_reconstr_gn2832.csv', sep ="\t")
print("read complete")

df["O_bar"] = np.nan

print("add cols complete")

for i in range(len(df)):
    if (df["O_bar"][i] != df["O_bar"][i]):
        Lat, Long, gn = df["Latitude"][i], df["Longitude"][i], df["Gamman"][i]
        located = df[(df["Longitude"]==Long)&(df["Latitude"] == Lat)&(df["Gamman"] == gn)]
        indices = located.index.tolist()
        obar = located["Oxygen"].mean()
        df.loc[indices, "O_bar"] = obar


print("averager complete")
print("Rows with no average value: ", (len(df)-len(df[["O_bar"]].dropna())))
df.to_csv("tracer_reconstr_gn2750_TSOavg_Longcircle.csv", index = False, sep = '\t')
