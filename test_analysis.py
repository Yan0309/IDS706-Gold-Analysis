import math

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning

from hw3_analysis import check_data_quality
from hw3_analysis import daily_returns
from hw3_analysis import load_data
from hw3_analysis import plot_gld_over_time
from hw3_analysis import plot_scatter_with_regression
from hw3_analysis import return_outliers
from hw3_analysis import train_gld_spx_model
from hw3_analysis import yearly_average
from hw3_analysis import yearly_correlation
from hw3_analysis import yearly_model_scores


def test_load_data(tmp_path):
    csv_path = tmp_path / "gold.csv"
    pd.DataFrame({"Date": ["2021-12-31", "2022-01-01"], "GLD": [180, 181]}).to_csv(
        csv_path, index=False
    )

    df = load_data(csv_path)

    assert df["Year"].tolist() == [2021, 2022]
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])


@pytest.mark.parametrize(
    ("csv_contents", "expected_exception"),
    [
        ("GLD\n180\n", KeyError),
        ("Date,GLD\nnot-a-date,180\n", ValueError),
    ],
    ids=["missing-date-column", "unparseable-date"],
)
def test_load_data_rejects_missing_or_unparseable_date(
    tmp_path, csv_contents, expected_exception
):
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(csv_contents)

    with pytest.raises(expected_exception):
        load_data(csv_path)


@pytest.mark.parametrize(
    ("test_df", "expected"),
    [
        (pd.DataFrame(columns=["A", "B"]), {"missing_values": 0, "duplicate_rows": 0}),
        (
            pd.DataFrame({"A": [1, None], "B": [None, 3]}),
            {"missing_values": 2, "duplicate_rows": 0},
        ),
        (
            pd.DataFrame({"A": [1, 1], "B": [4, 4]}),
            {"missing_values": 0, "duplicate_rows": 1},
        ),
    ],
)
def test_check_data_quality(test_df, expected):
    assert check_data_quality(test_df) == expected


def test_yearly_average():
    test_df = pd.DataFrame(
        {"Year": [2020, 2020, 2021, 2021], "GLD": [100, 200, 300, 400]}
    )

    result = yearly_average(test_df, ["GLD"])

    assert result.loc[2021, "GLD"] == 350
    assert result.loc[2020, "GLD"] == 150


def test_yearly_average_handles_missing_zero_negative_outlier_and_single_rows():
    test_df = pd.DataFrame(
        {
            "Year": [2020, 2020, 2020, 2021, 2021, 2022],
            "GLD": [0, -10, None, 1, 1001, 7],
        }
    )

    result = yearly_average(test_df, ["GLD"])

    assert result.loc[2020, "GLD"] == -5
    assert result.loc[2021, "GLD"] == 501
    assert result.loc[2022, "GLD"] == 7


def test_yearly_average_empty_input_returns_empty_result():
    test_df = pd.DataFrame(columns=["Year", "GLD"])

    assert yearly_average(test_df, ["GLD"]).empty


@pytest.mark.parametrize(
    "test_df",
    [pd.DataFrame({"GLD": [1]}), pd.DataFrame({"Year": [2020]})],
)
def test_yearly_average_missing_columns_raises_key_error(test_df):
    with pytest.raises(KeyError):
        yearly_average(test_df, ["GLD"])


def test_yearly_average_non_numeric_values_raise_type_error():
    test_df = pd.DataFrame({"Year": [2020, 2020], "GLD": ["high", "low"]})

    with pytest.raises(TypeError):
        yearly_average(test_df, ["GLD"])


def test_train_gld_spx_model():
    test_df = pd.DataFrame({"SPX": [1, 2, 3, 4], "GLD": [300, 500, 700, 900]})

    _, r2 = train_gld_spx_model(test_df)

    assert r2 > 0.99


def test_daily_returns_sorts_full_series_drops_first_row_and_preserves_year():
    test_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2021-01-04", "2020-12-30", "2021-01-01", "2020-12-31"]
            ),
            "Year": [2021, 2020, 2021, 2020],
            "SPX": [133.1, 100, 121, 110],
            "GLD": [266.2, 200, 242, 220],
        }
    )

    result = daily_returns(test_df)

    assert result["Date"].tolist() == pd.to_datetime(
        ["2020-12-31", "2021-01-01", "2021-01-04"]
    ).tolist()
    assert result["Year"].tolist() == [2020, 2021, 2021]
    assert result["SPX"].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert result["GLD"].tolist() == pytest.approx([0.1, 0.1, 0.1])


def test_shared_price_trend_can_produce_high_price_r2_and_low_return_r2():
    rng = np.random.default_rng(42)
    days = np.arange(1000, dtype=float)
    test_df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=len(days)),
            "Year": 2020,
            "SPX": 5000 + 20 * days + rng.normal(0, 250, len(days)),
            "GLD": 2500 + 10 * days + rng.normal(0, 250, len(days)),
        }
    )

    _, price_r2 = train_gld_spx_model(test_df)
    _, returns_r2 = train_gld_spx_model(daily_returns(test_df))

    assert price_r2 > 0.9
    assert returns_r2 < 0.1


def test_return_outliers_flags_only_obvious_spike_day():
    returns_df = pd.DataFrame(
        {
            "SPX": [1.0] * 19 + [100.0],
            "GLD": list(range(100, 120)),
        }
    )

    result = return_outliers(returns_df)

    assert result.to_numpy().sum() == 1
    assert result.loc[19, "SPX"]
    assert not result["GLD"].any()


def test_return_outliers_flags_nothing_without_spikes():
    returns_df = pd.DataFrame(
        {
            "SPX": list(range(1, 21)),
            "GLD": list(range(101, 121)),
        }
    )

    result = return_outliers(returns_df)

    assert not result.to_numpy().any()


def test_train_gld_spx_model_supports_zero_and_negative_values():
    test_df = pd.DataFrame({"SPX": [-2, 0, 2], "GLD": [-5, 1, 7]})

    model, r2 = train_gld_spx_model(test_df)

    assert model.predict(pd.DataFrame({"SPX": [-1, 1]})).tolist() == pytest.approx(
        [-2, 4]
    )
    assert r2 == 1


def test_train_gld_spx_model_single_row_has_undefined_r2():
    test_df = pd.DataFrame({"SPX": [10], "GLD": [200]})

    with pytest.warns(UndefinedMetricWarning, match=r"R\^2 score is not well-defined"):
        model, r2 = train_gld_spx_model(test_df)

    assert model.predict(pd.DataFrame({"SPX": [10]})).tolist() == [200]
    assert pd.isna(r2)


@pytest.mark.parametrize(
    "test_df",
    [
        pd.DataFrame(columns=["SPX", "GLD"]),
        pd.DataFrame({"SPX": [1, 2], "GLD": [None, 3]}),
        pd.DataFrame({"SPX": [1, 2], "GLD": ["low", "high"]}),
    ],
)
def test_train_gld_spx_model_rejects_empty_missing_or_invalid_values(test_df):
    with pytest.raises(ValueError):
        train_gld_spx_model(test_df)


@pytest.mark.parametrize(
    "test_df",
    [pd.DataFrame({"GLD": [1, 2]}), pd.DataFrame({"SPX": [1, 2]})],
)
def test_train_gld_spx_model_missing_columns_raises_key_error(test_df):
    with pytest.raises(KeyError):
        train_gld_spx_model(test_df)


@pytest.mark.parametrize("nan_column", ["SPX", "GLD"])
def test_train_gld_spx_model_rejects_nan_values(nan_column):
    test_df = pd.DataFrame({"SPX": [1.0, 2.0], "GLD": [3.0, 4.0]})
    test_df.loc[0, nan_column] = float("nan")

    with pytest.raises(ValueError):
        train_gld_spx_model(test_df)


def test_yearly_correlation_returns_known_correlations():
    test_df = pd.DataFrame(
        {
            "Year": [2020, 2020, 2021, 2021],
            "GLD": [1, 2, 1, 2],
            "EUR/USD": [2, 4, 4, 2],
        }
    )

    result = yearly_correlation(test_df, "GLD", "EUR/USD")

    assert result[2020] == pytest.approx(1.0)
    assert result[2021] == pytest.approx(-1.0)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_yearly_correlation_single_row_year_is_nan():
    test_df = pd.DataFrame(
        {"Year": [2020, 2020, 2021], "GLD": [1, 2, 4], "EUR/USD": [2, 4, 9]}
    )

    result = yearly_correlation(test_df, "GLD", "EUR/USD")

    assert math.isnan(result[2021])


def test_yearly_model_scores_returns_known_scores():
    test_df = pd.DataFrame(
        {
            "Year": [2020, 2020, 2021, 2021],
            "SPX": [1, 2, 1, 2],
            "GLD": [3, 5, 7, 5],
        }
    )

    assert yearly_model_scores(test_df) == {2020: 1.0, 2021: 1.0}


def test_yearly_model_scores_single_row_year_warns_and_returns_nan():
    test_df = pd.DataFrame(
        {"Year": [2020, 2020, 2021], "SPX": [1, 2, 3], "GLD": [3, 5, 7]}
    )

    with pytest.warns(UndefinedMetricWarning):
        result = yearly_model_scores(test_df)

    assert math.isnan(result[2021])


def test_plot_scatter_with_regression_writes_nonempty_png(tmp_path):
    test_df = pd.DataFrame({"SPX": [1, 2, 3], "GLD": [3, 4, 8]})
    model, _ = train_gld_spx_model(test_df)
    output_path = tmp_path / "scatter.png"

    plot_scatter_with_regression(test_df, model, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_gld_over_time_writes_nonempty_png(tmp_path):
    test_df = pd.DataFrame(
        {"Date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "GLD": [3, 4]}
    )
    output_path = tmp_path / "gold-over-time.png"

    plot_gld_over_time(test_df, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_full_pipeline():
    df = load_data("data/gold_data_2015_25.csv")
    assert len(df) == 2666
    assert "Year" in df.columns

    quality = check_data_quality(df)
    assert quality["missing_values"] == 0
    assert quality["duplicate_rows"] == 0

    yearly_avg = yearly_average(df, ["GLD", "EUR/USD"])
    assert len(yearly_avg) > 0

    _, r2 = train_gld_spx_model(df)
    assert 0 <= r2 <= 1
