# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:42:37 2023

@author: Kyle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import requests
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
from scipy import stats


# %% Load all the data

# Data can be downloaded at: 
#    https://www.pac.dfo-mpo.gc.ca/science/charts-cartes/obs-app/observed-eng.aspx?StationID=07795

Pt_Atkn = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Pt_Atkn_all_obs.csv'))
Pt_Atkn = Pt_Atkn.drop(['Unnamed: 0'], axis=1)
# Pt_Atkn = Pt_Atkn.rename(columns={'TIME_TAG UTC (Z+0)': 'datetime', 'ENCODER1': 'water_level1', 'ENCODER2': 'water_level2', 'PREDICTION': 'pred'})
Pt_Atkn['datetime'] = pd.to_datetime(Pt_Atkn['datetime'])
Pt_Atkn.set_index('datetime', inplace=True)

# Create new column called 'water_level' that is a copy of 'water_level1', but 
# equals 'water_level2' when 'water_level1' is NaN.
Pt_Atkn['water_level'] = Pt_Atkn['water_level1'].fillna(Pt_Atkn['water_level2'])
Pt_Atkn = Pt_Atkn.drop(['water_level1', 'water_level2'], axis=1)

Pt_Atkn['diff'] = Pt_Atkn['water_level'] - Pt_Atkn['pred']

# Localize timezone to make datetime tz aware
Pt_Atkn = Pt_Atkn.tz_localize(tz='UTC')




# Station info: Bellingham International Airport (KBLI) 
# This can be downloaded at: https://mesowest.utah.edu/
# Units: 
    # timezone: UTC
    # wind_speed: miles/hour
    # wind_direction: degrees
    # wind gust: miles/hour
    # Precip: inches
    # sea_level_pressure: inHg (inches of mercury)
BLI_obs = pd.read_csv(os.path.join(os.path.dirname(__file__), 'KBLI_all_obs.csv'))
BLI_obs['datetime'] = pd.to_datetime(BLI_obs['Date_Time'])
BLI_obs = BLI_obs.drop(['Date_Time'], axis=1)
BLI_obs.set_index('datetime', inplace=True)

# # convert inHg --> hPa -- 
# # hPa = inHg / 0.02953
BLI_obs['SLP'] = round((BLI_obs['SLP'] / 0.02953), 2)


# %% Plotting water level/SLP time series

fig, ax = plt.subplots(1, 1)

ax.plot(Pt_Atkn['water_level'], 'b')
ax.plot(Pt_Atkn['pred'], 'k.')

ax1 = ax.twinx()
ax1.plot(BLI_obs['SLP'], 'r', linewidth=3)
ax1.invert_yaxis()
plt.ticklabel_format(useOffset=False, axis='y')

# %% Storm surge comparisons with different weather variables.


# criteria_wind = BLI_obs['peak_wind_speed'][(BLI_obs['peak_wind_direction'] > 90) & (BLI_obs['peak_wind_direction'] < 270)]
zero_line = np.zeros(Pt_Atkn['diff'].shape)

fig, ax = plt.subplots(1,1)

# ax.plot(criteria_wind, 'g.', label='Max wind between 90-270 degrees in direction (MPH)')
# ax.plot(BLI_obs['pressure_change'].dropna(), 'g.', label='Rapid changes in pressure')
ax.plot(-BLI_obs['SLP'])
ax.plot([], [], 'r', label='Storm surge when total water levels >= 5m')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Max southerly wind (MPH)')
ax1 = ax.twinx()
ax1.plot(Pt_Atkn.index, zero_line, 'k', linewidth=1)
ax1.plot(Pt_Atkn['diff'].dropna(), 'r', linewidth=0.5)
# ax1.plot(Pt_Atkn['diff'][Pt_Atkn['water_level'] >= 5], 'r.', label='Storm surge when total water levels >= 5m')
ax1.set_ylabel('Storm surge (m)')


# %% Regression analysis SLP

# Reindex SLP obs; interpolate nearest values to match Pt. Atkinson diff obs
SLP = BLI_obs['SLP'].dropna().reindex(Pt_Atkn['diff'].dropna().index, method='nearest')
surge = Pt_Atkn['diff'].dropna()

fig, ax = plt.subplots(1,1)
# fig.set_size_inches(14, 8)

ax.plot(SLP, surge, 'k.', markersize=5)

pt1_line = np.ones(SLP.shape) * 0.1
ax.plot(SLP, pt1_line, 'b')
pt2_line = np.ones(SLP.shape) * 0.2
ax.plot(SLP, pt2_line, 'b')
pt3_line = np.ones(SLP.shape) * 0.3
ax.plot(SLP, pt3_line, 'b')
pt4_line = np.ones(SLP.shape) * 0.4
ax.plot(SLP, pt4_line, 'b')
pt5_line = np.ones(SLP.shape) * 0.5
ax.plot(SLP, pt5_line, 'b')
pt6_line = np.ones(SLP.shape) * 0.6
ax.plot(SLP, pt6_line, 'b')



x = SLP.values.reshape(-1, 1)
y = surge.values

#add constant to predictor variable
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view and store model summary in 'results'
print(model.summary())
results = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]

y_int = results['coef'][0]
slope = results['coef'][1]
r_sq = model.rsquared

# model = LinearRegression().fit(x, y)
# r_sq = model.score(x, y)
# b_0 = model.intercept_
# b_1 = model.coef_

ax.set_title('Predicted/Observed difference in water height compared with SLP ' + '($R^{2}$ = ' + str(round(r_sq, 2)) + ')')
ax.set_ylabel('Water level difference (ft)')
ax.set_xlabel('Sea level pressure (hPa)')
plt.ticklabel_format(useOffset=False, axis='x')

reg_y = (slope * x[:,1]) + y_int
ax.plot(x[:,1], reg_y, 'r', label = '$y = ' + str(round(slope, 2)) + 'x + ' + str(round(y_int, 2)) + '$',
          linewidth=2)
ax.legend()



# plot residuals
# mu, std = stats.norm.fit(model.resid)
# fig2, ax2 = plt.subplots()
# sns.histplot(x=model.resid, ax=ax2, stat="density", linewidth=0, kde=True)
# ax2.set(title="Distribution of residuals (SLP vs storm surge)", xlabel="residual")
# xmin, xmax = plt.xlim() # the maximum x values from the histogram above
# x = np.linspace(xmin, xmax, 100) # generate some x values
# p = stats.norm.pdf(x, mu, std) # calculate the y values for the normal curve
# sns.lineplot(x=x, y=p, color="orange", ax=ax2)



# %% Regression model prediction

# New DataFrame incorporating all the independent and dependent variables to be used (size = 52704 = surge.size)
# independent variables: 
    # SLP
    # wind
    # precip
# dependent variable
    # surge

# surge
Pt_Atkn = Pt_Atkn.drop(Pt_Atkn.loc[Pt_Atkn.index < pd.Timestamp('2016-10-01 00:00:00', tz='UTC')].index)
surge = Pt_Atkn['diff'].dropna()
surge.rename('surge', inplace=True)

# SLP
# Reindex SLP obs; interpolate nearest values to match Pt. Atkinson diff obs
SLP = BLI_obs['SLP'].dropna().reindex(surge.index, method='nearest')

# # wind
# wind = BLI_obs[['wind_speed', 'wind_gust', 'peak_wind_speed']].dropna(how='all')
# wind['max_wind'] = wind.max(axis=1, skipna= True)
# wind = wind['max_wind']
# # Reindex wind obs; interpolate nearest values to match Pt. Atkinson diff obs
# wind = wind.reindex(surge.index, method='nearest')
# wind.rename('wind', inplace=True)

# # precip
# precip_6hr = BLI_obs['6hr_precip']
# # interpolate missing values 
# int_precip = precip_6hr.interpolate(method='time')
# # Reindex precip obs; interpolate nearest values to match Pt. Atkinson diff obs
# precip = int_precip.reindex(surge.index, method='nearest').fillna(0)
# precip.rename('precip', inplace=True)


# New DataFrame to be used for multiple linear regression
delta = pd.concat([surge, SLP], axis=1)

# Regression
X = delta['SLP']
X = sm.add_constant(X)
y = delta['surge']

result = sm.OLS(y, X).fit()
print(result.summary())

# mu, std = stats.norm.fit(result.resid)
# fig2, ax2 = plt.subplots()
# sns.histplot(x=result.resid, ax=ax2, stat="density", linewidth=0, kde=True)
# ax2.set(title="Distribution of residuals (SLP vs storm surge)", xlabel="residual")
# xmin, xmax = plt.xlim() # the maximum x values from the histogram above
# x = np.linspace(xmin, xmax, 100) # generate some x values
# p = stats.norm.pdf(x, mu, std) # calculate the y values for the normal curve
# sns.lineplot(x=x, y=p, color="orange", ax=ax2)

# # Variance Inflation Factor calculation to measure multcolinearity
# vif = pd.DataFrame()
# vif['ind_variables'] = X.columns[1:] # remove the constant
# vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])[1:]]

def pred_interval(model, SLP, sig_lvl):
    # model: OLS model.
    # SLP: input sea level pressure value
    # sig_lvl: level of significance.
    X = [1, SLP] # put input value into correct format
    predictions = model.get_prediction(X)
    frame = predictions.summary_frame(alpha=sig_lvl)
    return frame['mean'][0], frame['obs_ci_lower'][0], frame['obs_ci_upper'][0]

new_SLP_value = float(input("SLP value in hPa: "))

best_guess, PI_lower, PI_upper = pred_interval(result, new_SLP_value, 0.05)

print("\n95% confidence the true storm surge value is between " + str(PI_lower.round(3)) + "m - " + str(PI_upper.round(3)) + "m.\n")
print("Best guess is " + str(best_guess.round(3)) + 'm.')    



# Create an array of realistic SLP values (800 - 1100)hPa
real_SLP = np.arange(900, 1051)
x = sm.add_constant(real_SLP)
pred = result.get_prediction(x)
frame = pred.summary_frame(alpha=0.05)

# add realistic SLP values to output DataFrame (frame)
frame['SLP_values'] = real_SLP
frame.set_index('SLP_values', inplace=True)

# write 'frame' to csv file
# frame.to_csv('C:/Users/Kyle/Documents/Summer Projects 2023 - WxNet/Website Dev/marbzgrop.github.io/Delta.csv')
