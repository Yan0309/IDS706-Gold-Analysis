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


def _per_year(df, func):
    return {
        year: func(group)
        for year, group in df.groupby("Year", sort=False)
    }


def yearly_correlation(df, col1, col2):
    """Calculate each year's correlation between two columns.

    Returns:
        A mapping from year to correlation.
    """
    return _per_year(df, lambda group: group[col1].corr(group[col2]))


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
    return _per_year(df, lambda group: train_gld_spx_model(group)[1])


def daily_returns(df, columns=("SPX", "GLD")):
    """Calculate returns across the full date-sorted series."""
    result = df.sort_values("Date").loc[:, ["Date", "Year", *columns]].copy()
    result[list(columns)] = result[list(columns)].pct_change()
    return result.dropna(subset=list(columns)).reset_index(drop=True)


def return_outliers(returns_df, columns=("SPX", "GLD"), threshold=4):
    """Return a boolean frame marking returns beyond the z-score threshold."""
    values = returns_df[list(columns)]
    z_scores = (values - values.mean()) / values.std()
    return z_scores.abs() > threshold


def plot_scatter_with_regression(
    df,
    model,
    output_path,
    title="The relationship between spx and gld",
    xlabel="SPX index",
    ylabel="Gold Price",
):
    """Save an SPX-versus-GLD scatter plot with the fitted regression line."""
    X = df[["SPX"]]
    y = df["GLD"]
    plt.figure()
    plt.scatter(X, y)
    plt.plot(X, model.predict(X), color="red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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
    returns_df = daily_returns(df)
    returns_model, returns_r2 = train_gld_spx_model(returns_df)
    print(f"Overall price R²: {r2:.3f}; overall returns R²: {returns_r2:.3f}")
    print(yearly_model_scores(df))
    yearly_returns_r2 = {
        year: round(score, 3)
        for year, score in yearly_model_scores(returns_df).items()
    }
    print(f"Yearly returns R²: {yearly_returns_r2}")

    outlier_masks = return_outliers(returns_df)
    for column in ("SPX", "GLD"):
        dates = returns_df.loc[
            outlier_masks[column], "Date"
        ].dt.strftime("%Y-%m-%d").tolist()
        print(f"{column} return outliers (|z| > 4): {len(dates)} days: {dates}")

    outlier_days = outlier_masks.any(axis=1)
    _, returns_without_outliers_r2 = train_gld_spx_model(
        returns_df.loc[~outlier_days]
    )
    print(
        "Returns R² without outlier days: "
        f"{returns_without_outliers_r2:.3f}"
    )

    plot_scatter_with_regression(df, model, "figures/spx_vs_gld_scatter.png")
    plot_scatter_with_regression(
        returns_df,
        returns_model,
        "figures/returns_scatter.png",
        title="Daily returns: SPX vs GLD",
        xlabel="SPX daily return",
        ylabel="GLD daily return",
    )
    plot_gld_over_time(df, "figures/gld_vs_time.png")


if __name__ == "__main__":
    main()
