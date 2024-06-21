import requests
import pandas as pd
from datetime import datetime, timedelta


############ Station IDs ############


def get_wx_station_id(station_code):
    """
    Map ICAO codes to NCEI station identifiers (AWSBAN QID)
    NCEI station identifiers can be found manually at https://www.ncei.noaa.gov/access/search/data-search/global-hourly

    Parameters:
        station_code (str): International Civil Aviation Organization (ICAO) code; airport ID

    Returns:
        str: NCEI station identifier (AWSBAN QID)
    """
    url = "https://raw.githubusercontent.com/kyleanders-on/kyleanders-on.github.io/main/NCEI_ID.json"
    response = requests.get(url)
    id_map = pd.json_normalize(response.json()["stationNames"])
    station_name = id_map.loc[id_map["qid"] == f"ICAO:{station_code}"][
        "preferredName"
    ].iloc[0]
    NCEI_id = (
        id_map.loc[id_map["preferredName"] == station_name]["qid"]
        .iloc[0]
        .replace("AWSBAN:", "")
    )
    return NCEI_id


def get_tide_station_id(station_name):
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


############ Sea level pressure (SLP) raw data collection ############


def gather_SLP_for_multiple_years(start_year, end_year, NCEI_id, http_success_codes):
    """
    Gather SLP data between October 1 to April 1 for multiple years.

    Parameters:
        start_year (int): The start year of the data retrieval.
        end_year (int): The end year of the data retrieval.
        NCEI_id (str): NCEI station identifier (AWSBAN QID).
        http_success_codes (range): HTTP response status codes success class (200-299).

    Returns:
        list: A list containing JSON data with SLP observations for each year.
    """
    data = []
    current_year = start_year
    while current_year < end_year:
        current_end = current_year + 1
        data.append(fetch_SLP(current_year, current_end, NCEI_id, http_success_codes))
        current_year += 1

    return data


def fetch_SLP(start_year, end_year, id, http_success_codes):
    """
    Request hourly SLP (Sea Level Pressure) observations from the NCEI database (Integrated Surface Dataset).
    Limit data requests to the date range October 1 - April 1.

    Parameters:
        start_year (int): The start year of the data retrieval.
        end_year (int): The end year of the data retrieval.
        id (str): NCEI station identifier (AWSBAN QID).
        http_success_codes (range): HTTP response status codes success class (200-299).

    Returns:
        list: A list containing JSON data with SLP observations for a single season.
    """
    base_url = "https://www.ncei.noaa.gov/access/services/data/v1"
    station_id = id  # AWSBAN Qualified ID
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


############ Tidal prediction and observation data collection ############


def gather_tidal_data_for_multiple_years(
    start_year, end_year, station_id, tide_product, http_success_codes
):
    """
    Gather tidal data between October 1 to April 1 for multiple years.

    Parameters:
        start_year (int): The start year of the data retrieval.
        end_year (int): The end year of the data retrieval.
        station_id (str): The NOAA station ID corresponding to the provided tidal station name.
        tide_product (str): The type of tide data to request (e.g., "water_level" or "predictions").
        http_success_codes (range): HTTP response status codes success class (200-299).

    Returns:
        list: A list containing JSON data with water level observations or base tide prediction data for multiple years.
    """
    data = []
    current_year = start_year
    while current_year < end_year:
        start_date = datetime(current_year, 10, 1)
        end_date = datetime(current_year + 1, 4, 2)
        data.extend(
            fetch_tide_data(
                start_date, end_date, station_id, tide_product, http_success_codes
            )
        )
        current_year += 1

    return data


def fetch_tide_data(start_date, end_date, station_id, tide_product, http_success_codes):
    """
    Gathers water level and tide data from NOAA CO-OPS API for a specific station.

    Parameters:
        start_date (datetime): Start date for data retrieval.
        end_date (datetime): End date for data retrieval.
        station_id (str): The station ID for the tide data.
        tide_product (str): The type of tide data to request (e.g., "water_level" or "predictions").

    Returns:
        list: A list containing JSON data with water level observations or base tide prediction data for a single season.
    """
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    product = tide_product
    datum = "NAVD"
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
