import numpy as np
import pandas as pd

# df = pd.read_csv('tracer_reconstr_ALL_TSavg_Longcircle.csv', sep ="\t")
df = pd.read_csv('tracer_reconstr_gn2832.csv', sep ="\t")
df2 = df[(df['Longitude'])]
