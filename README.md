# Tidal Flooding Analysis in the Salish Sea

The effects of tidal flooding in the Salish Sea have been evident to vulnerable communities in the region for decades. Following a particularly severe flooding event in late December 2022, which notably impacted the South Park neighborhood of Seattle, Washington, adjacent to the Duwamish Watershed, and residential communities along Boundary Bay in the city of Delta, British Columbia, there is an urgent need to develop additional tools for forecasting storm surge. This project aims to create a linear regression model to be used as a storm surge forecasting tool to assist vulnerable communities in the Salish Sea region in preparing for and mitigating the impacts of tidal flooding events.

## Demo

Check out the live version of the tool [here](https://kyleanders-on.github.io/).

## Concept

`tidal_flooding_US.py` and `tidal_flooding_CAN.py` are designed to access and process observed and modelled weather and tidal datasets. After performing multiple linear regression analysis with several predictor weather variables as well as upstream discharge for the Duwamish Watershed, it was observed that sea level pressure had the largest coefficient of determination by a significant margin. Consequently, the other independent variables were excluded from the final regression model.

Results of the regression analysis have revealed that there is moderate-strong correlation between mean sea-level pressure (MSLP) and storm surge. For this project, storm surge is estimated based on the difference between astronomical tide predictions and observed water levels. Astronomical tide predictions in the United States are produced by NOAA's Center for Operational Oceanographic Products and Services (CO-OPS) and by the Canadian Hydrographic Service (CHS) in Canada.

This tool will use a linear regression model to predict a 95% prediction interval for storm surge, based on new MSLP values. This will allow forecasters to use numerical weather prediction to estimate a range of likely water levels to assist in the preparation for impactful flooding events.

## Objective

The December 2022 flooding event underscored the importance of evaluating existing forecasting methods, highlighting the urgent necessity for an accurate and dependable tool to predict storm surges in the Salish Sea. By developing this forecasting tool, I hope to empower communities with useful information that can aid in planning for, and ultimately minimizing, the adverse effects of tidal flooding.

While this project's primary focus is on the Duwamish Watershed and the Boundary Bay area, its methodology can be applied to other regions within the Salish Sea. However, it is important to acknowledge that certain limitations may affect the tool's forecasting accuracy, such as data availability and the complexity of meteorological and tidal factors involved.
