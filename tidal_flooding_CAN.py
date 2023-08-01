# %% [markdown]
# Import necessary packages

# %%
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy import stats


# %% Map ICAO codes to NCEI station identifiers (AWSBAN QID)
# Still a work in progress. AWSBAN QIDs are available at https://www.ncei.noaa.gov/access/homr/services/station/simple/names/
# but that file is too large to justify this mapping feature.
wx_station_code = "KBFI"
url = (
    "https://www.ncei.noaa.gov/access/homr/services/station/search?qid=ICAO:"
    + wx_station_code
)
response = requests.get(url)
id_map = pd.json_normalize(
    response.json()["stationCollection"]["stations"], record_path=["identifiers"]
)


# %% Map tidal station names to station ID

# If station name is unknown, you can browse available stations at https://tidesandcurrents.noaa.gov/


def get_station_id(station_name):
    url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    response = requests.get(url)
    id_map = pd.json_normalize(response.json()["stations"])
    id_map.set_index(id_map["name"], inplace=True)
    id_map.drop(columns=["name"], inplace=True)

    try:
        station_id = id_map.loc[station_name]["id"]
        return station_id
    except KeyError:
        print(f"Error: Station '{station_name}' not found in the NOAA database.")
        return None


tidal_station_name = "Seattle"
tidal_station_id = get_station_id(tidal_station_name)


# %% Initialize date range. Dates spanning only October 1 - April 1 will be evaluated.
start_year = 2020
end_year = 2023

# %% [markdown]
# Request SLP data from NCEI database (Integrated Surface Dataset)


# %% Hourly SLP observations
def fetch_SLP(start_year, end_year):
    base_url = "https://www.ncei.noaa.gov/access/services/data/v1"
    station_id = "72793524234"  # AWSBAN Qualified ID
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
    else:
        print(
            f"Failed to fetch data for {start} to {end}. Status Code: {response.status_code}"
        )

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
def fetch_water_level_data(start_date, end_date, station_id):
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    product = "water_level"
    datum = "MLLW"
    units = "english"
    time_zone = "GMT"
    format_type = "json"

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


def gather_data_for_multiple_years(start_year, end_year, station_id):
    data = []
    current_year = start_year
    while current_year < end_year:
        start_date = datetime(current_year, 10, 1)
        end_date = datetime(current_year + 1, 4, 2)
        data.extend(fetch_water_level_data(start_date, end_date, station_id))
        current_year += 1

    return data


water_level_data = gather_data_for_multiple_years(
    start_year, end_year, tidal_station_id
)

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
def fetch_tides(start_date, end_date, station_id):
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    product = "predictions"
    datum = "MLLW"
    units = "english"
    time_zone = "GMT"
    format_type = "json"

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


def gather_data_for_multiple_years(start_year, end_year, station_id):
    data = []
    current_year = start_year
    while current_year < end_year:
        start_date = datetime(current_year, 10, 1)
        end_date = datetime(current_year + 1, 4, 2)
        data.extend(fetch_tides(start_date, end_date, station_id))
        current_year += 1
    return data


tide_prediction_data = gather_data_for_multiple_years(
    start_year, end_year, tidal_station_id
)

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

# plot best fit line
reg_line = model.predict(X)
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)
ax.plot(X["SLP"].values, y.values, "k.")
ax.plot(X["SLP"].values, reg_line.values, "r")
ax.set_xlabel(f"{wx_station_code} Sea Level Pressure (hPa)")
ax.set_ylabel(f"{tidal_station_name} Storm surge estimate (ft)")
ax.set_title(
    "Observed water level/base tide prediction difference vs sea level pressure",
)

fig.show()

# Compare residuals distribution to a normal curve
mu, std = stats.norm.fit(model.resid)
fig2, ax2 = plt.subplots()
fig2.set_size_inches(10, 7)
sns.histplot(x=model.resid, ax=ax2, stat="density", linewidth=0, kde=False)
sns.kdeplot(
    x=model.resid, ax=ax2, label="Kernel Density Estimation", linewidth=2.5, color="g"
)
ax2.set(title="Distribution of residuals (SLP vs storm surge)", xlabel="residual")
xmin, xmax = plt.xlim()  # the maximum x values from the histogram above
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
sns.lineplot(
    x=x, y=p, color="orange", ax=ax2, label="Normal Distribution", linewidth=2.5
)

fig2.show()

# plot residuals vs leverage
norm_resid = model.get_influence().resid_studentized_internal
lev = model.get_influence().hat_matrix_diag
Cooks_distance = model.get_influence().cooks_distance[0]

# plot_lm = plt.figure()
fig3, ax3 = plt.subplots()
fig3.set_size_inches(10, 7)
plt.scatter(lev, norm_resid, alpha=0.5)
sns.regplot(
    x=lev,
    y=norm_resid,
    scatter=False,
    ci=False,
    lowess=True,
    line_kws={"color": "red", "lw": 1, "alpha": 0.8},
)

ax3.set_title("Residuals vs Leverage")
ax3.set_xlabel("Leverage")
ax3.set_ylabel("Standardized Residuals")

leverage_top_3 = np.flip(np.argsort(Cooks_distance), 0)[:3]
for i in leverage_top_3:
    ax3.annotate(i, xy=(lev[i], norm_resid[i]))

fig3.show()
# %% [markdown]
# Regression model prediction output


# %%
def pred_interval(result, SLP, sig_lvl):
    # result: OLS model.
    # SLP: input sea level pressure value
    # sig_lvl: level of significance.
    X = [1, SLP]  # put input value into correct format
    predictions = result.get_prediction(X)
    frame = predictions.summary_frame(alpha=sig_lvl)
    return frame["mean"][0], frame["obs_ci_lower"][0], frame["obs_ci_upper"][0]


# new_SLP_value = float(input("SLP value in hPa: "))

# best_guess, PI_lower, PI_upper = pred_interval(model, new_SLP_value, 0.05)

# print(
#     "\n95% confidence the true storm surge value is between "
#     + str(PI_lower.round(3))
#     + "ft - "
#     + str(PI_upper.round(3))
#     + "ft.\n"
# )
# print("Best guess is " + str(best_guess.round(3)) + "ft.")

# Create an array of realistic SLP values (900-1050)hPa
real_SLP = np.arange(900, 1051)
x = sm.add_constant(real_SLP)
pred = model.get_prediction(x)
frame = pred.summary_frame(alpha=0.05)

# add realistic SLP values to output DataFrame
frame["SLP_values"] = real_SLP
frame.set_index("SLP_values", inplace=True)

# %%
