import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy import stats

# comparing Elliott Bay water level obs to Marginal Way Bridge water level obs
# already able to get Elliott Bay water level obs in data_collection.py
# data request url example: https://nwis.waterservices.usgs.gov/nwis/iv/?format=json&site=12113415&startDT=2023-10-01T00:00&endDT=2024-04-02T00:00
# required url parameters: format=JSON, siteID, start_date, end_date


def gather_Duwamish_for_multiple_years(
    start_year, end_year, USGS_id, http_success_codes
):
    """
    Gather water level observation data from the E Marginal Way Bridge tidal gage between October 1 to April 1 for multiple years.

    Parameters:
        start_year (int): The start year of the data retrieval.
        end_year (int): The end year of the data retrieval.
        USGS_id (str): USGS site ID number for the Duwamish River at E Marginal Bridge monitoring site
        http_success_codes (range): HTTP response status codes success class (200-299)

    Returns:
        list: A list containing JSON data with Duwmaish water level observations for each year.
    """
    data = []
    current_year = start_year
    while current_year < end_year:
        current_end = current_year + 1
        data.append(
            fetch_water_levels(current_year, current_end, USGS_id, http_success_codes)
        )
        current_year += 1

    return data


def fetch_water_levels(start_year, end_year, id, http_success_codes):
    """
    Request 15 minute water level observations from the USGS database.
    Limit data requests to the date range October 1 - April 1.

    Parameters:
        start_year (int): The start year of the data retrieval.
        end_year (int): The end year of the data retrieval.
        id (str): USGS site ID number for the Duwamish River at E Marginal Bridge monitoring site
        http_success_codes (range): HTTP response status codes success class (200-299)

    Returns:
        list: A list containing JSON data with water level observations.
    """
    base_url = "https://nwis.waterservices.usgs.gov/nwis/iv/?"
    station_id = id  # USGS site ID
    start = datetime(start_year, 10, 1)
    end = datetime(end_year, 4, 2)
    format_type = "json"

    API_arguments = {
        "format": format_type,
        "site": station_id,
        "startDT": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endDT": end.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    data = []
    response = requests.get(base_url, params=API_arguments)
    if response.status_code in http_success_codes:
        data.append(response.json())
    else:
        raise ValueError(
            f"Failed to fetch data for {start} to {end}. Status Code: {response.status_code}"
        )

    return data[0]["value"]["timeSeries"][1]["values"][0]["value"]


def process_Duwamish_json(duwamish_json):
    """
    Convert nested json to a Pandas DataFrame.

    Parameters:
        duwamish_json (list): A list containing JSON data with Duwmaish water level observations for each year.

    Returns:
        pandas.DataFrame: structured DataFrame containing Duwamish Waterway water level data
    """

    # flatten data: list of 4 inner lists, each containing dictionaries containing each observation
    flattened_data = [dictionary for sublist in duwamish_json for dictionary in sublist]
    df = pd.DataFrame(flattened_data)
    df["date_time"] = pd.to_datetime(df["dateTime"], utc=True)
    df.set_index("date_time", inplace=True)
    df.drop(columns=["dateTime", "qualifiers"], inplace=True)
    df.rename(columns={"value": "water_level"}, inplace=True)
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")
    return df


def tidal_comparison(duwamish_water_levels, elliott_bay_obs):
    """
    Compute them mean difference between Elliott Bay and Duwamish Waterway water level observations.

    The difference is computed on the third quartile (Lower 75% of the data removed) only to focus
    on high tides (low tides are more notably different between the two sites).

    Parameters:
        duwamish_water_levels (pandas.DataFrame): Marginal Way Bridge water level observations Oct 1-April 1.
        elliott_bay_obs (pandas.DataFrame): Elliott Bay water level observations Oct 1-April 1.

    Returns:
        float: mean difference between Elliott Bay and Duwamish Waterway water level observations.
    """

    # duwamish_water_levels: 15 minute frequency
    # elliott_bay_water_levels: 6 minute frequency

    # upsample duwamish_water_levels to a 6 minute frequency to match Elliott Bay obs
    duwamish_resampled = duwamish_water_levels.reindex(
        elliott_bay_obs.index, method="nearest"
    )

    merged_tidal_obs = pd.merge(
        elliott_bay_obs,
        duwamish_resampled,
        on="date_time",
        suffixes=("_elliott_bay", "_duwamish"),
    )

    threshold_elliott = merged_tidal_obs["water_level_elliott_bay"].quantile(0.75)
    threshold_duwamish = merged_tidal_obs["water_level_duwamish"].quantile(0.75)
    high_tides = merged_tidal_obs[
        (merged_tidal_obs["water_level_elliott_bay"] > threshold_elliott)
        & (merged_tidal_obs["water_level_duwamish"] > threshold_duwamish)
    ]

    mean_diff = (
        high_tides["water_level_duwamish"].mean()
        - high_tides["water_level_elliott_bay"].mean()
    )

    return mean_diff
