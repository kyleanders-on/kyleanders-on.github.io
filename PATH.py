# %% [markdown]
# Import necessary packages
# %%
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# %% [markdown]
# Request SLP data from Mesonet database
# %%
API_token = "b29add930aa84e39bb1fa38bac04165b"

station_id = "KEWR"
API_root = "https://api.synopticdata.com/v2/stations/timeseries"

start_year, end_year = "2022", "2023"
wx_variable = "sea_level_pressure"

API_arguments = {
    "token": API_token,
    "stid": station_id,
    "start": start_year + "10010000",
    "end": end_year + "04020000",
    "vars": wx_variable,
}


req = requests.get(API_root, params=API_arguments)
json_data = req.json()

parsed_data = json_data["STATION"][0]["OBSERVATIONS"]
KEWR = pd.DataFrame(parsed_data)
KEWR.set_index(KEWR["date_time"], inplace=True)


# %% [markdown]
# Request tidal data from NOAA CO-OPS
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


# Water level data spanning October 1 to April 2 for
# the specified range of years
start_year, end_year = 2019, 2023
water_level_data = gather_data_for_multiple_years(start_year, end_year)


Batt_obs = pd.DataFrame(water_level_data)
Batt_obs.rename(columns={"t": "date_time", "v": "water_level"}, inplace=True)
Batt_obs["date_time"] = pd.to_datetime(Batt_obs["date_time"])
Batt_obs.set_index(Batt_obs["date_time"], inplace=True)
Batt_obs.drop(["date_time", "s", "f", "q"], axis=1, inplace=True)
Batt_obs = Batt_obs.tz_localize(tz="UTC")
Batt_obs["water_level"] = Batt_obs["water_level"].astype(float)

# %% [markdown]
# Process data
# %%
