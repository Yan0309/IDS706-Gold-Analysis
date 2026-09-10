import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# load the file into a DataFrame
df = pd.read_csv("data/gold_data_2015_25.csv") 

# change date from str to datetime
df["Date"] = pd.to_datetime(df["Date"])

# get year out of the date
df["Year"] = df["Date"].dt.year

# check the head
print(df.head())

# check data types and non-null counts for each column
print(df.info())

# check the stats for columns
print(df.describe())

# check duplicated 
print(sum(df.duplicated()))

# check the average between gld price and EUR/USD yearly
yearly_avg = df.groupby("Year")[["GLD", "EUR/USD"]].mean()
print(yearly_avg)

# check the correlation between these two columns in general
print(df[["GLD","EUR/USD"]].corr())

# check the correlation between these two columns yearly
years = df["Year"].unique()
for y in years:
    year_df = df[df["Year"] == y]
    print(y,year_df[["GLD","EUR/USD"]].corr())
    

X = df[["SPX"]]
y = df["GLD"]

# train the model of SPX and GLD
model = LinearRegression()
model.fit(X, y)

# try to see coefficient, intercept, and the R^2 between SPX and GLD
print(model.coef_)
print(model.intercept_)
print(model.score(X, y))

# to check do they really have the relationship yearly
for yr in years:
    year_df = df[df["Year"] == yr]
    X_year = year_df[["SPX"]]
    y_year = year_df["GLD"]
    
    model_year = LinearRegression()
    model_year.fit(X_year, y_year)
    print(yr,model_year.score(X_year, y_year))

# make scatter plot for the relationship between spx and gld    
plt.scatter(X, y)
plt.xlabel("SPX index")
plt.ylabel("Gold Price")
plt.title("The relationship between spx and gld")
plt.plot(X, model.predict(X), color="red")
plt.savefig("figures/spx_vs_gld_scatter.png")

# make the plot for the change of gold price yearly
plt.figure()
plt.plot(df["Date"], df["GLD"])
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.title("Gold price over time (2015-2025)")
plt.savefig("figures/gld_vs_time.png")