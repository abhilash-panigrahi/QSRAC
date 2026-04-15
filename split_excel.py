import pandas as pd

file_path = "02-14-2018.csv"

df = pd.read_csv(file_path)

n = len(df)
chunk = n // 8

df[:chunk].to_csv("part1.csv", index=False)
df[chunk:2*chunk].to_csv("part2.csv", index=False)
df[2*chunk:3*chunk].to_csv("part3.csv", index=False)
df[3*chunk:4*chunk].to_csv("part4.csv", index=False)
df[4*chunk:5*chunk].to_csv("part5.csv", index=False)
df[5*chunk:6*chunk].to_csv("part6.csv", index=False)
df[6*chunk:7*chunk].to_csv("part7.csv", index=False)
df[7*chunk:].to_csv("part8.csv", index=False)