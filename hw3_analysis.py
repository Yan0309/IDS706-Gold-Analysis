import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def load_data(filepath):
    """Load gold prices and add a datetime-derived Year column."""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    return df


def check_data_quality(df):
    """Count missing values and duplicate rows.

    Returns:
        Counts under "missing_values" and "duplicate_rows".
    """
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    return {"missing_values": missing, "duplicate_rows": duplicates}


def yearly_average(df, columns):
    """Calculate yearly means for the selected columns."""
    return df.groupby("Year")[columns].mean()


def yearly_correlation(df, col1, col2):
    """Calculate each year's correlation between two columns.

    Returns:
        A mapping from year to correlation.
    """
    years = df["Year"].unique()
    result = {}
    for yr in years:
        year_df = df[df["Year"] == yr]
        corr = year_df[col1].corr(year_df[col2])
        result[yr] = corr
    return result


def train_gld_spx_model(df):
    """Fit a GLD-on-SPX linear regression model.

    Returns:
        The fitted model and its training R-squared score.
    """
    X = df[["SPX"]]
    y = df["GLD"]
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    return model, r2


def yearly_model_scores(df):
    """Fit yearly GLD-on-SPX models and return training R-squared scores."""
    years = df["Year"].unique()
    scores = {}
    for yr in years:
        year_df = df[df["Year"] == yr]
        X_year = year_df[["SPX"]]
        y_year = year_df["GLD"]
        model_year = LinearRegression()
        model_year.fit(X_year, y_year)
        scores[yr] = model_year.score(X_year, y_year)
    return scores


def plot_scatter_with_regression(df, model, output_path):
    """Save an SPX-versus-GLD scatter plot with the fitted regression line."""
    X = df[["SPX"]]
    y = df["GLD"]
    plt.figure()
    plt.scatter(X, y)
    plt.plot(X, model.predict(X), color="red")
    plt.xlabel("SPX index")
    plt.ylabel("Gold Price")
    plt.title("The relationship between spx and gld")
    plt.savefig(output_path)
    plt.close()


def plot_gld_over_time(df, output_path):
    """Save a line plot of GLD prices over time."""
    plt.figure()
    plt.plot(df["Date"], df["GLD"])
    plt.xlabel("Date")
    plt.ylabel("Gold Price")
    plt.title("Gold price over time (2015-2025)")
    plt.savefig(output_path)
    plt.close()


def main():
    df = load_data("data/gold_data_2015_25.csv")

    print(df.head())
    print(df.info())
    print(df.describe())

    quality = check_data_quality(df)
    print(quality)

    yearly_avg = yearly_average(df, ["GLD", "EUR/USD"])
    print(yearly_avg)

    print(df[["GLD", "EUR/USD"]].corr())
    print(yearly_correlation(df, "GLD", "EUR/USD"))

    model, r2 = train_gld_spx_model(df)
    print(model.coef_)
    print(model.intercept_)
    print(r2)
    print(yearly_model_scores(df))

    plot_scatter_with_regression(df, model, "figures/spx_vs_gld_scatter.png")
    plot_gld_over_time(df, "figures/gld_vs_time.png")


if __name__ == "__main__":
    main()
