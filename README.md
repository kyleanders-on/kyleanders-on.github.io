# Project Background

The effects of tidal flooding in the Salish Sea have been evident to vulnerable communities in the region for decades. Following a particularly severe flooding event in late December 2022, which notably impacted the South Park neighborhood of Seattle, Washington, adjacent to the Duwamish Waterway, and residential communities along Boundary Bay in the city of Delta, British Columbia, this project was started with the goal of developing additional tools for forecasting storm surge. The outcome of this project is a linear regression model that produces a prediction interval of storm surge values when given a forecasted sea-level pressure (SLP) value. 

## Demo

Check out the live version of the tool [here](https://kyleanders-on.github.io/).

## Concept

The program is designed to access and process observed and modeled weather and tidal datasets using several public APIs. Initially, multiple linear regression analysis was performed with several predictor weather variables as well as upstream discharge in the Green-Duwamish River Watershed. These initial predictor weather variables were SLP, rainfall, and wind. The analysis revealed that among the predictor variables, only SLP exhibited a significant correlation with the response variable, storm surge. Consequently, the model was refined by excluding the other predictor variables that did not contribute significantly to the explanatory power of the model. Additionally, due to the relationship between SLP, wind, and rainfall, the Variance Inflation Factor (VIF) was used to quantify multicollinearity among these predictors and the observed degree of multicollinearity remained within an acceptable range.

Results of the regression analysis have revealed that there is meaningful correlation between SLP and storm surge. For this project, storm surge is estimated based on the difference between astronomical tide predictions and observed water levels. Astronomical tide predictions in the United States are produced by NOAA's Center for Operational Oceanographic Products and Services (CO-OPS) and by the Canadian Hydrographic Service (CHS) in Canada.

This tool will use the linear regression model to predict a 95% prediction interval for storm surge, based on forecasted SLP values. This will allow forecasters to use numerical weather prediction to estimate a range of likely future water levels to assist in the preparation for impactful flooding events.

## Application to Other Sites in the Region

While this project's primary focus is on the Duwamish Waterway and the Boundary Bay area, its methodology can be applied to other regions within the Salish Sea. However, it is important to acknowledge that certain limitations may affect the tool's forecasting accuracy, such as data availability and unique underwater/coastal topography.
