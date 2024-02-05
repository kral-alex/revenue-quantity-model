import os
import pandas

PATH = "../Data/statsfinal.csv"
df = pandas.read_csv(PATH)
df.drop(columns=['Unnamed: 0'], inplace=True)

print(df)

df['P-P1'] = df['S-P1'] / df['Q-P1']
df['P-P2'] = df['S-P2'] / df['Q-P2']
df['P-P3'] = df['S-P3'] / df['Q-P3']
df['P-P4'] = df['S-P4'] / df['Q-P4']

print(df.groupby('P-P1').groups.keys())
print(df.groupby('P-P2').groups.keys())
print(df.groupby('P-P3').groups.keys())
print(df.groupby('P-P4').groups.keys())


