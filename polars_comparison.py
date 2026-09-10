import time
import pandas as pd
import polars as pl

# pandas read file time 
start = time.time()
df_pandas = pd.read_csv("data/gold_data_2015_25.csv")
end = time.time()
print("Pandas read_csv time:", end - start)

# polars read file time
start = time.time()
df_polars = pl.read_csv("data/gold_data_2015_25.csv")
end = time.time()
print("Polars read_csv time:", end - start)

# change date from str to datetime 
df_pandas["Date"] = pd.to_datetime(df_pandas["Date"])

# get year out of the date
df_pandas["Year"] = df_pandas["Date"].dt.year

# select column, change this column to datetime, name it as Date
df_polars = df_polars.with_columns(
    pl.col("Date").str.to_datetime().alias("Date"))

# extract year from Date
df_polars = df_polars.with_columns(
    pl.col("Date").dt.year().alias("Year"))

# pandas outcome time
start = time.time()
yearly_avg_pandas = df_pandas.groupby("Year")[["GLD", "EUR/USD"]].mean()
end = time.time()
print("Pandas groupby time:", end - start)

# polar outcome time
start = time.time()
yearly_avg_polars = df_polars.group_by("Year").agg(pl.col("GLD").mean(), pl.col("EUR/USD").mean())
end = time.time()
print("Polars groupby time:", end - start)