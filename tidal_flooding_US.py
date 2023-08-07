import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression


# Initialize date range. Dates spanning only October 1 - April 1 will be evaluated.
start_year = 2022
end_year = 2023

# HTTP response status codes success class (200-299)
http_success_codes = range(200, 300)

# Map ICAO codes to NCEI station identifiers (AWSBAN QID)
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


def get_station_id(station_name):
    """
    Get the NOAA CO-OPS station ID for a given tidal station name.
    If station name is unknown, you can browse available stations at https://tidesandcurrents.noaa.gov/.

    Parameters:
        station_name (str): The name of the tidal station.

    Returns:
        str: The NOAA station ID corresponding to the provided tidal station name.

    Raises:
        ValueError: If the station name is not found in the NOAA database.
    """
    url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    response = requests.get(url)
    id_map = pd.json_normalize(response.json()["stations"]).set_index("name")

    try:
        station_id = id_map.loc[station_name]["id"]
        return station_id
    except KeyError as error:
        raise ValueError(
            f'Error: Station "{station_name}" not found in the NOAA database.'
        ) from error


tidal_station_name = "Seattle"
tidal_station_id = get_station_id(tidal_station_name)


def fetch_SLP(start_year, end_year):
    """
    Request hourly SLP (Sea Level Pressure) observations from the NCEI database (Integrated Surface Dataset).
    Limit data requests to the date range October 1 - April 1.

    Parameters:
        start_year (int): The start year of the data retrieval.
        end_year (int): The end year of the data retrieval.

    Returns:
        list: A list containing JSON data with SLP observations.
    """
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
    if response.status_code in http_success_codes:
        data.append(response.json())
    else:
        raise ValueError(
            f"Failed to fetch data for {start} to {end}. Status Code: {response.status_code}"
        )

    return data


def gather_SLP_for_multiple_years(start_year, end_year):
    """
    Gather SLP data between October 1 to April 1 for multiple years.

    Parameters:
        start_year (int): The start year of the data retrieval.
        end_year (int): The end year of the data retrieval.

    Returns:
        list: A list containing JSON data with SLP observations for each year.
    """
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
SLP_data.set_index("DATE", inplace=True)
SLP_data = SLP_data.tz_localize(tz="UTC")
SLP_data.drop(
    columns=["REPORT_TYPE", "QUALITY_CONTROL", "STATION", "SOURCE"],
    inplace=True,
)
SLP_data["SLP"].replace(",", ".", regex=True, inplace=True)
SLP_data["SLP"] = pd.to_numeric(SLP_data["SLP"], errors="coerce") / 10
SLP_data.loc[SLP_data["SLP"] == 9999.99] = np.nan


def fetch_tide_data(start_date, end_date, station_id, tide_product):
    """
    Gathers water level and tide data from NOAA CO-OPS API for a specific station.

    Parameters:
        start_date (datetime): Start date for data retrieval.
        end_date (datetime): End date for data retrieval.
        station_id (str): The station ID for the tide data.
        tide_product (str): The type of tide data to request (e.g., "water_level" or "predictions").

    Returns:
        list: A list containing json data with water level observations or base tide prediction data.
    """
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    product = tide_product
    datum = "MLLW"
    units = "english"
    time_zone = "GMT"
    format_type = "json"

    interval = timedelta(days=31)
    current_date = start_date
    data = []

    while current_date <= end_date:
        start = current_date.strftime("%Y%m%d %H:%M")
        if product == "water_level":
            end = (current_date + interval).strftime("%Y%m%d %H:%M")
        else:
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

        # Adjust the end_date in the last request
        if current_date + interval > end_date:
            params["end_date"] = end_date.strftime("%Y%m%d %H:%M")

        response = requests.get(base_url, params=params)
        if response.status_code in http_success_codes:
            if product == "water_level":
                data.extend(response.json()["data"])
            else:
                data.extend(response.json()["predictions"])
        else:
            raise ValueError(
                f"Failed to fetch data for {start} to {end}. Status Code: {response.status_code}"
            )

        if product == "water_level":
            current_date += interval + timedelta(minutes=6)
        else:
            current_date = end_date + interval

    return data


def gather_data_for_multiple_years(start_year, end_year, station_id, tide_product):
    """
    Gather tidal data between October 1 to April 1 for multiple years.

    Parameters:
        start_year (int): The start year of the data retrieval.
        end_year (int): The end year of the data retrieval.
        tide_product (str): The type of tide data to request (e.g., "water_level" or "predictions").

    Returns:
        list: A list containing json data with water level observations or base tide prediction data.
    """
    data = []
    current_year = start_year
    while current_year < end_year:
        start_date = datetime(current_year, 10, 1)
        end_date = datetime(current_year + 1, 4, 2)
        data.extend(fetch_tide_data(start_date, end_date, station_id, tide_product))
        current_year += 1

    return data


def process_json(start_year, end_year, tidal_station_id, tide_product):
    """
    Convert list containing json data to Pandas DataFrame with relavent information.

    Parameters:
        start_year (int): The start year of the data retrieval.
        end_year (int): The end year of the data retrieval.
        tidal_station_id (str): The NOAA station ID.
        tide_product (str): The type of tide data to request (e.g., "water_level" or "predictions").

    Returns:
        Pandas DataFrame: A DataFrame with relavent tidal data.
    """
    data = gather_data_for_multiple_years(
        start_year, end_year, tidal_station_id, tide_product
    )
    df = pd.DataFrame(data)
    df.rename(columns={"t": "date_time", "v": "water_level"}, inplace=True)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df.set_index("date_time", inplace=True)
    df = df.tz_localize(tz="UTC")
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")

    if tide_product == "water_level":
        df.drop(["s", "f", "q"], axis=1, inplace=True)

    return df


# 6-minute water level obs. NOAA CO-OPS API limits data retrievals to 31 days for this product
tide_product = "water_level"
water_level_obs = process_json(start_year, end_year, tidal_station_id, tide_product)

# 6-minute base tide predictions from NOAA CO-OPS
tide_product = "predictions"
tide_pred = process_json(start_year, end_year, tidal_station_id, tide_product)


# Regression analysis and data visualization
# Perform Ordinary Least Squares (OLS) linear regression and fit the model
SLP = SLP_data["SLP"].dropna()
surge = (water_level_obs - tide_pred).dropna().reindex(SLP.index, method="nearest")


X = SLP
X = sm.add_constant(X)
y = surge["water_level"]
model = sm.OLS(y, X).fit()

# Print summary statistics
print(model.summary())


# plot best fit line
reg_line = model.predict(X)
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)
ax.plot(X["SLP"].values, y.values, "k.")
ax.plot(X["SLP"].values, reg_line.values, "r", label="OLS model fit")
ax.set_xlabel(f"{wx_station_code} Sea Level Pressure (hPa)")
ax.set_ylabel(f"{tidal_station_name} Storm surge estimate (ft)")
ax.set_title(
    "Observed water level/base tide prediction difference vs sea level pressure",
)


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


# plot residuals vs leverage
norm_resid = model.get_influence().resid_studentized_internal
lev = model.get_influence().hat_matrix_diag
Cooks_distance = model.get_influence().cooks_distance[0]

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


# Regression model prediction output
# Create an array of realistic SLP values (900-1050 hPa)
real_SLP = np.arange(900, 1051)
x = sm.add_constant(real_SLP)
pred = model.get_prediction(x)
model_output = pred.summary_frame(alpha=0.05)

# Add realistic SLP values to output DataFrame
model_output["SLP_values"] = real_SLP
model_output.set_index("SLP_values", inplace=True)

ax.plot(model_output["obs_ci_lower"], "b--")
ax.plot(model_output["obs_ci_upper"], "b--", label="95% prediction interval")
ax.legend()
plt.show()
