import pandas as pd

from hw3_analysis import load_data
from hw3_analysis import check_data_quality
from hw3_analysis import yearly_average
from hw3_analysis import train_gld_spx_model

def test_load_data():
    df = load_data("data/gold_data_2015_25.csv")
    assert len(df) == 2666
    assert "Year" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])




def test_check_data_quality():
    test_df = pd.DataFrame({
    "A": [1, 2, 2],
    "B": [4, 5, 5]
    })
    result = check_data_quality(test_df)
    assert result["missing_values"] == 0
    assert result["duplicate_rows"] == 1


def test_yearly_average():
    test_df = pd.DataFrame({
    "Year": [2020, 2020, 2021, 2021],
    "GLD": [100, 200, 300, 400]
    })
    result = yearly_average(test_df, ["GLD"])
    result.loc[2021, "GLD"]
    assert result.loc[2021, "GLD"] == 350
    assert result.loc[2020, "GLD"] == 150
    
def test_train_gld_spx_model():
    test_df = pd.DataFrame({
        "SPX": [1, 2, 3, 4],
        "GLD": [300, 500, 700, 900]
    })
    model, r2 = train_gld_spx_model(test_df)
    assert r2 > 0.99
    
def test_full_pipeline():
    # Step 1: load real data
    df = load_data("data/gold_data_2015_25.csv")
    assert len(df) > 0
    assert "Year" in df.columns

    # Step 2: check data quality
    quality = check_data_quality(df)
    assert quality["missing_values"] == 0
    assert quality["duplicate_rows"] == 0

    # Step 3: yearly average
    yearly_avg = yearly_average(df, ["GLD", "EUR/USD"])
    assert len(yearly_avg) > 0

    # Step 4: train model
    model, r2 = train_gld_spx_model(df)
    assert 0 <= r2 <= 1