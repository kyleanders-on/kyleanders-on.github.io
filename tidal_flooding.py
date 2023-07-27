# %% [markdown]
# Import necessary packages

# %%
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm


# %% Map ICAO codes to NCEI station identifiers
station_code = "KEWR"
url = (
    "https://www.ncei.noaa.gov/access/homr/services/station/search?qid=ICAO:"
    + station_code
)
response = requests.get(url)
id_map = pd.json_normalize(
    response.json()["stationCollection"]["stations"], record_path=["identifiers"]
)

# %% Initialize date range. Dates spanning only October 1 - April 1 will be evaluated.
start_year = 2019
end_year = 2020

# %% [markdown]
# Request SLP data from NCEI database (Integrated Surface Dataset)


# %% Hourly SLP observations
def fetch_SLP(start_year, end_year):
    base_url = "https://www.ncei.noaa.gov/access/services/data/v1"
    station_id = "72502014734"  # KEWR AWSBAN Qualified ID
    wx_variable = "SLP"
    start = datetime(start_year, 10, 1)
    end = datetime(end_year, 4, 2)
    format_type = "json"

    API_arguments = {
        "dataset": "global-hourly",
        "stations": station_id,
        "startDate": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endDate": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataTypes": wx_variable,
        "format": format_type,
    }

    data = []
    response = requests.get(base_url, params=API_arguments)
    if 200 <= response.status_code < 300:
        data.append(response.json())

    return data


def gather_SLP_for_multiple_years(start_year, end_year):
    data = []
    current_year = start_year
    while current_year < end_year:
        current_end = current_year + 1
        data.append(fetch_SLP(current_year, current_end))
        current_year += 1

    return data


SLP_data = gather_SLP_for_multiple_years(start_year, end_year)

SLP_data = pd.json_normalize(SLP_data, record_path=[0])
SLP_data["DATE"] = pd.to_datetime(SLP_data["DATE"])
SLP_data.set_index(SLP_data["DATE"], inplace=True)
SLP_data = SLP_data.tz_localize(tz="UTC")
SLP_data.drop(
    columns=["REPORT_TYPE", "DATE", "QUALITY_CONTROL", "STATION", "SOURCE"],
    inplace=True,
)
SLP_data["SLP"].replace(",", ".", regex=True, inplace=True)
SLP_data["SLP"] = SLP_data["SLP"].astype(float) / 10
SLP_data.loc[SLP_data["SLP"] == 9999.99] = np.nan

# %% [markdown]
# Request water level observations from NOAA CO-OPS


# %% 6-minute water level obs. NOAA CO-OPS API limits data retrievals to 31 days for this product
def fetch_water_level_data(start_date, end_date):
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    product = "water_level"
    datum = "MLLW"
    units = "english"
    time_zone = "GMT"
    format_type = "json"
    station_id = "8518750"  # The Battery (south end of Manhatten)

    interval = timedelta(days=31)
    current_date = start_date
    data = []

    while current_date <= end_date:
        start = current_date.strftime("%Y%m%d %H:%M")
        end = (current_date + interval).strftime("%Y%m%d %H:%M")

        params = {
            "product": product,
            "datum": datum,
            "units": units,
            "time_zone": time_zone,
            "format": format_type,
            "begin_date": start,
            "end_date": end,
            "station": station_id,
        }

        # Adjust the end_date in the last request
        if current_date + interval > end_date:
            params["end_date"] = end_date.strftime("%Y%m%d %H:%M")

        response = requests.get(base_url, params=params)
        if 200 <= response.status_code < 300:
            data.extend(response.json()["data"])
        else:
            print(
                f"Failed to fetch data for {start} to {end}. Status Code: {response.status_code}"
            )

        current_date += interval + timedelta(days=1)

    return data


def gather_data_for_multiple_years(start_year, end_year):
    data = []
    current_year = start_year
    while current_year < end_year:
        start_date = datetime(current_year, 10, 1)
        end_date = datetime(current_year + 1, 4, 2)
        data.extend(fetch_water_level_data(start_date, end_date))
        current_year += 1

    return data


water_level_data = gather_data_for_multiple_years(start_year, end_year)

water_level_obs = pd.DataFrame(water_level_data)
water_level_obs.rename(columns={"t": "date_time", "v": "water_level"}, inplace=True)
water_level_obs["date_time"] = pd.to_datetime(water_level_obs["date_time"])
water_level_obs.set_index(water_level_obs["date_time"], inplace=True)
water_level_obs.drop(["date_time", "s", "f", "q"], axis=1, inplace=True)
water_level_obs = water_level_obs.tz_localize(tz="UTC")
water_level_obs["water_level"] = water_level_obs["water_level"].astype(float)

# %% [markdown]
# Request base tide predictions from NOAA CO-OPS


# %% 6-minute base tide predictions from NOAA CO-OPS
def fetch_tides(start_date, end_date):
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    product = "predictions"
    datum = "MLLW"
    units = "english"
    time_zone = "GMT"
    format_type = "json"
    station_id = "8518750"  # The Battery (south end of Manhatten)

    current_date = start_date
    data = []

    start = current_date.strftime("%Y%m%d %H:%M")
    end = (end_date).strftime("%Y%m%d %H:%M")
    params = {
        "product": product,
        "datum": datum,
        "units": units,
        "time_zone": time_zone,
        "format": format_type,
        "begin_date": start,
        "end_date": end,
        "station": station_id,
    }
    response = requests.get(base_url, params=params)
    if 200 <= response.status_code < 300:
        data.extend(response.json()["predictions"])
    else:
        print(
            f"Failed to fetch data for {start} to {end}. Status Code: {response.status_code}"
        )

    return data


def gather_data_for_multiple_years(start_year, end_year):
    data = []
    current_year = start_year
    while current_year < end_year:
        start_date = datetime(current_year, 10, 1)
        end_date = datetime(current_year + 1, 4, 2)
        data.extend(fetch_tides(start_date, end_date))
        current_year += 1
    return data


tide_prediction_data = gather_data_for_multiple_years(start_year, end_year)

tide_pred = pd.DataFrame(tide_prediction_data)
tide_pred.rename(columns={"t": "date_time", "v": "water_level"}, inplace=True)
tide_pred["date_time"] = pd.to_datetime(tide_pred["date_time"])
tide_pred.set_index(tide_pred["date_time"], inplace=True)
tide_pred.drop(["date_time"], axis=1, inplace=True)
tide_pred = tide_pred.tz_localize(tz="UTC")
tide_pred["water_level"] = tide_pred["water_level"].astype(float)

# %% [markdown]
# Process data

# %%

SLP = SLP_data["SLP"].dropna()
surge = (water_level_obs - tide_pred).dropna().reindex(SLP.index, method="nearest")

X = SLP
X = sm.add_constant(X)
y = surge["water_level"]

model = sm.OLS(y, X).fit()
print(model.summary())
reg_line = model.predict(X)

fig, ax = plt.subplots()
ax.plot(X["SLP"].values, y.values, "k.")
ax.plot(X["SLP"].values, reg_line.values, "r")

# %%
