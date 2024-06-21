import pandas as pd
import numpy as np


def SLP_processing(SLP_data):
    """
    Convert list containing JSON data to Pandas DataFrame with relavent information.

    Parameters:
        SLP_data (list): A list containing JSON data with SLP for multiple years.

    Returns:
        pandas.DataFrame: A DataFrame with relavent SLP data.
    """
    SLP_data = pd.json_normalize(SLP_data, record_path=[0])
    SLP_data["DATE"] = pd.to_datetime(SLP_data["DATE"])
    SLP_data.set_index("DATE", inplace=True)
    SLP_data = SLP_data.tz_localize(tz="UTC")
    SLP_data.drop(
        columns=["REPORT_TYPE", "QUALITY_CONTROL", "STATION", "SOURCE"],
        inplace=True,
    )
    SLP_data.replace({"SLP": ","}, {"SLP": "."}, regex=True, inplace=True)
    SLP_data["SLP"] = pd.to_numeric(SLP_data["SLP"], errors="coerce") / 10
    SLP_data.loc[SLP_data["SLP"] == 9999.99] = np.nan

    return SLP_data


def process_tidal_json(tide_product, data):
    """
    Convert list containing JSON data to Pandas DataFrame with relavent information.

    Parameters:
        tide_product (str): The type of tide data to request (e.g., "water_level" or "predictions").
        data (list): A list containing JSON data with water levels for multiple years.

    Returns:
        pandas.DataFrame: A DataFrame with relavent tidal data.
    """
    df = pd.DataFrame(data)
    df.rename(columns={"t": "date_time", "v": "water_level"}, inplace=True)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df.set_index("date_time", inplace=True)
    df = df.tz_localize(tz="UTC")
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")

    if tide_product == "water_level":
        df.drop(["s", "f", "q"], axis=1, inplace=True)

    return df
