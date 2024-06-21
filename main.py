from data_collection import (
    get_wx_station_id,
    get_tide_station_id,
    gather_SLP_for_multiple_years,
    gather_tidal_data_for_multiple_years,
)
from data_processing import SLP_processing, process_tidal_json
from regression_analysis import reg_model, model_prediction_output
from duwamish_site_comparison import (
    gather_Duwamish_for_multiple_years,
    process_Duwamish_json,
    tidal_comparison,
)
from config import START_YEAR, END_YEAR, HTTP_SUCCESS_CODES, SOUTH_PARK_DUWAMISH_ID


def main():

    ########## Get station ID information ##########
    # Airport weather sensor
    wx_station_code = "KBFI"
    NCEI_id = get_wx_station_id(wx_station_code)

    # Regression model reference tide gauge
    tidal_station_name = "Seattle"
    tidal_station_id = get_tide_station_id(tidal_station_name)

    ########## Collect and process tidal data (observations and predictions) ##########
    # 15-minute Duwamish Waterway (Marginal Way Bridge) water level obs; resampled to 6-minute frequency
    duwamish_water_levels_unfiltered = gather_Duwamish_for_multiple_years(
        START_YEAR, END_YEAR, SOUTH_PARK_DUWAMISH_ID, HTTP_SUCCESS_CODES
    )
    duwamish_water_levels = process_Duwamish_json(duwamish_water_levels_unfiltered)

    # 6-minute Elliott Bay water level obs.
    tide_product = "water_level"
    tidal_obs_unprocessed = gather_tidal_data_for_multiple_years(
        START_YEAR, END_YEAR, tidal_station_id, tide_product, HTTP_SUCCESS_CODES
    )
    elliott_bay_obs = process_tidal_json(tide_product, tidal_obs_unprocessed)

    # mean difference between Elliott Bay and Duwamish Waterway water levels
    mean_difference = tidal_comparison(duwamish_water_levels, elliott_bay_obs)

    # 6-minute base tide predictions from NOAA CO-OPS
    tide_product = "predictions"
    tidal_pred_unprocessed = gather_tidal_data_for_multiple_years(
        START_YEAR, END_YEAR, tidal_station_id, tide_product, HTTP_SUCCESS_CODES
    )
    tide_pred = process_tidal_json(tide_product, tidal_pred_unprocessed)

    ########## Collect and process SLP data ##########
    SLP_data_uprocessed = gather_SLP_for_multiple_years(
        START_YEAR, END_YEAR, NCEI_id, HTTP_SUCCESS_CODES
    )
    SLP_data = SLP_processing(SLP_data_uprocessed)

    ########## Linear regression analysis ##########
    model = reg_model(SLP_data, elliott_bay_obs, tide_pred)[0]
    model_pred = model_prediction_output(model, mean_difference)
    print(model_pred)
    # Write model prediction output to csv file
    formatted_station_name = tidal_station_name.replace(" ", "_")
    model_pred.to_csv(f"./{formatted_station_name}.csv")


if __name__ == "__main__":
    main()
