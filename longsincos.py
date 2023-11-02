import pandas as pd
import numpy as np

df = pd.read_csv('tracer_reconstr_gn2750_TSavg.csv', sep ='\t')
# df = pd.read_csv('tracer_reconstr_gn2832_TSavg.csv', sep ='\t')
print("read complete")
df['Long_x'] = np.cos(np.radians(df['Longitude']))
df['Long_y'] = np.sin(np.radians(df['Longitude']))
print("computation complete")
df.to_csv("tracer_reconstr_gn2750_TSavg_Longcircle.csv", index = False, sep = '\t')
print('save complete')
print(df.head())
