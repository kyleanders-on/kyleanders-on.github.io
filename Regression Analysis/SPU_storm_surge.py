# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:00:42 2023

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
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from scipy import stats
import pickle


# %% [1] reading in data

target_dir = os.getcwd()
relative_path = '\\Seattle Public Utilities - Duwamish Storm Surge\\File Concatenation\\'
path = target_dir + relative_path

# Station info: Duwamish River at E Marginal Way Bridge Gauge
# This can be downloaded at: 
    # https://waterdata.usgs.gov/monitoring-location/12113415/#parameterCode=00065&startDT=2022-11-01&endDT=2023-03-29
# Units
    # timezone: UTC
    # water level: feet
# Duwamish_obs = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Duwamish_all_obs.csv'))
# Duwamish_obs['datetime'] = pd.to_datetime(Duwamish_obs['datetime'])
# Duwamish_obs.set_index('datetime', inplace=True)
# # Convert PDT --> UTC
# Duwamish_obs = Duwamish_obs.tz_localize(tz='US/Pacific', ambiguous=True)
# Duwamish_obs = Duwamish_obs.tz_convert('UTC')

# # Drop obs before October 1, 2019 and after April 1, 2023
# Duwamish_obs = Duwamish_obs.drop(Duwamish_obs.loc[Duwamish_obs.index < pd.Timestamp('2019-10-01 00:00:00', tz='UTC')].index)
# Duwamish_obs = Duwamish_obs.drop(Duwamish_obs.loc[Duwamish_obs.index > pd.Timestamp('2023-04-01 23:45:00', tz='UTC')].index)


# Station info: Green River at Purification Plant near Palmer, WA
# This can be downloaded at: 
    # https://waterdata.usgs.gov/monitoring-location/12106700/#parameterCode=00060&startDT=2022-10-01&endDT=2023-04-01
# Units
    # timezone: UTC
    # discharge: cubic feet per second
# Palmer_plant = pd.read_csv(os.path.join(os.path.dirname(__file__), 'nwis.waterservices.usgs.gov_palmer.csv'))
# Palmer_plant['datetime'] = pd.to_datetime(Palmer_plant['datetime'])
# Palmer_plant.set_index('datetime', inplace=True)
# # Convert PDT --> UTC
# Palmer_plant = Palmer_plant.tz_localize(tz='US/Pacific', ambiguous=True)
# Palmer_plant = Palmer_plant.tz_convert('UTC')


# Station info: Seattle Waterfront NOAA Tide Station (station id: 9447130)
# This can be downloaded at: https://tidesandcurrents.noaa.gov/noaatideannual.html?id=9447130
# Units
    # timezone: UTC
    # Datum: NAVD 88
# Seattle_pred = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Seattle_pred_all_obs.csv'))


# Seattle_pred['datetime'] = pd.to_datetime(Seattle_pred['datetime'])
# Seattle_pred.set_index('datetime', inplace=True)

# # Localize timezone to make datetime tz aware
# Seattle_pred = Seattle_pred.tz_localize(tz='UTC')


# Station info: Seattle Waterfront NOAA Tide Station (station id: 9447130)
# This can be downloaded at: https://tidesandcurrents.noaa.gov/waterlevels.html?id=9447130
# Units
    # timezone: UTC
    # Datum: NAVD 88
    
Elliott_Bay = pd.read_csv(path + 'Elliott Bay obs\\Elliott_Bay_all_obs.csv',
                          parse_dates=[['Date', 'Time']])
Elliott_Bay['datetime'] = pd.to_datetime(Elliott_Bay['Date_Time'])
Elliott_Bay.set_index('datetime', inplace=True)
Elliott_Bay = Elliott_Bay.drop(['Date_Time'], axis=1)
Elliott_Bay['surge'] = Elliott_Bay['obs'] - Elliott_Bay['pred']

# Localize timezone to make datetime tz aware
Elliott_Bay = Elliott_Bay.tz_localize(tz='UTC')
    

# Station info: Boeing Field-King County International Airport (KBFI) 
# This can be downloaded at: https://mesowest.utah.edu/
# Units: 
    # timezone: UTC
    # wind_speed: miles/hour
    # wind_direction: degrees
    # wind gust: miles/hour
    # Precip: inches
    # sea_level_pressure: inHg (inches of mercury)
BFI_obs = pd.read_csv(path + 'KBFI obs\\KBFI_all_obs.csv')
BFI_obs['datetime'] = pd.to_datetime(BFI_obs['Date_Time'])
BFI_obs = BFI_obs.drop(['Date_Time'], axis=1)
BFI_obs.set_index('datetime', inplace=True)
# BFI_obs = BFI_obs[BFI_obs['SLP'] != 'None']
BFI_obs['SLP'] = BFI_obs.SLP.astype(float)

# convert inHg --> hPa -- 
# hPa = inHg / 0.02953
BFI_obs['SLP'] = round((BFI_obs['SLP'] / 0.02953), 2)



# %% [2] plotting - SLP/Water level time series
sig_criteria = np.ones(Duwamish_obs.shape[0]) * 10

# plt.style.use('classic')
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(14, 8)

ax1.plot(Duwamish_obs.index, sig_criteria, 'r--', label='10 ft above NAVD', linewidth=3)
# ax1.plot(Duwamish_obs['water_level'], '#211f1f', linewidth=1)
daily_max_water = Duwamish_obs.resample("D").max()['water_level']

# last value does not include the high tide for the day so drop it
daily_max_water = daily_max_water[:-1]

ax1.plot(daily_max_water, '#298f39', label='Daily max water level', linewidth=5)
# ax1.plot(Duwamish_obs.resample("D").mean()['water_level'], '#3333d6', label='Daily mean', linewidth=3)

ax1.set_title('Duwamish River Water Height (ft) at E Marginal Way Bridge (SLP Overlaid)')
ax1.set_ylabel('Gauge Height in feet (NAVD 88)')
ax1.set_xlabel('Date')
ax1.plot([], [], '#1f1d1a', label='SLP')
ax1.legend()

ax2 = ax1.twinx()
ax2.plot(BFI_obs['SLP'], '#1f1d1a', label='SLP', linewidth=1)
ax2.set_ylabel('Sea Level Pressure (hPa)')
ax2.invert_yaxis()

# Prevent scientific notation
plt.ticklabel_format(useOffset=False, axis='y')



# %% [3] plotting - Difference between water level obs and tidal predictions

# obs = Duwamish_obs.resample('D').max()['water_level']
# pred = Seattle_pred.resample('D').max()['Pred(Ft)']
pred = Seattle_pred.loc[Seattle_pred['High/Low'] == 'H']['Pred(Ft)']
obs = Duwamish_obs.resample('12h').max()['water_level']
obs = obs.reindex(pred.index, method='ffill')

diff = obs - pred
# first entry mismatched so droping
diff = diff.iloc[1:]

# plt.plot(Duwamish_obs['water_level'], label='original data')
# plt.plot(obs, linewidth=3, label='12hr max heights')
# plt.legend()

zero_line = np.zeros(diff.shape[0])

fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(14, 8)

# ax1.plot(diff, 'k.')


# # ax1.plot(obs, '#388f52', marker='o', linestyle='-', label='Observations', linewidth=2)
# # ax1.plot(pred, '#f49858', marker='o', linestyle='-', label='Tide Prediction', linewidth=2)
ax1.plot(-diff, '#0af00d', marker='h', linestyle='', label='difference')
ax1.plot(diff.index, zero_line, 'k')
# ax1.set_ylim(-3, 3)
ax1.set_ylabel('Water level in feet')
ax1.set_xlabel('Date')
ax1.set_title('Difference between observed water level and tide predictions with SLP overlaid')
plt.fill_between(diff.index, -diff, where = (diff >= 0), color='r', alpha=0.3)
plt.fill_between(diff.index, -diff, where = (diff < 0), color='b', alpha=0.3)
ax1.plot([], [], '#1f1d1a', label='SLP')
ax1.legend()

ax2 = ax1.twinx()
ax2.plot(BFI_obs['SLP'], '#1f1d1a', label='SLP', linewidth=2)
ax2.set_ylabel('Sea Level Pressure (hPa)')

# Prevent scientific notation
plt.ticklabel_format(useOffset=False, axis='y')


# %% Regression Analysis SLP

# f(x) = b_1x + b_0
# where b_1 = slope, b_0 = y-intercept

# SLP_daily_max = BFI_obs.resample('6h').max()['SLP']

SLP_semidiurnal_min = BFI_obs.resample('12h').min()['SLP']
SLP_semidiurnal_min = SLP_semidiurnal_min.reindex(diff.index, method='ffill')


# Duwamish_obs_daily_max = Duwamish_obs.resample('6h').max()['water_level']
# Duwamish_obs_daily_min = Duwamish_obs.resample('6h').min()['water_level']



fig, ax1 = plt.subplots(1,1)
fig.set_size_inches(14, 8)

ax1.plot(SLP_semidiurnal_min, diff, 'k.')
# ax1.plot(SLP_daily_max, Duwamish_obs_daily_min, 'r.')

# SLP_semidiurnal_min = SLP_semidiurnal_min.drop(SLP_semidiurnal_min.index[186])
# diff = diff.drop(diff.index[186])

x = SLP_semidiurnal_min.values.reshape(-1, 1)
y = diff.values

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

ax1.set_title('Predicted/Observed difference in water height compared with semidiurnal SLP minimum ' + '($R^{2}$ = ' + str(round(r_sq, 2)) + ')')
ax1.set_ylabel('Water level difference (ft)')
ax1.set_xlabel('Minimum Semidiurnal Sea level pressure (hPa)')
plt.ticklabel_format(useOffset=False, axis='x')

reg_y = (slope * x[:,1]) + y_int
ax1.plot(x[:,1], reg_y, 'r:', label = '$y = ' + str(round(slope, 2)) + 'x + ' + str(round(y_int, 2)) + '$',
          linewidth=2)
ax1.legend()

# plot residuals
mu, std = stats.norm.fit(model.resid)
fig2, ax2 = plt.subplots()
sns.histplot(x=model.resid, ax=ax2, stat="density", linewidth=0, kde=True)
ax2.set(title="Distribution of residuals (SLP vs storm surge)", xlabel="residual")
xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x = np.linspace(xmin, xmax, 100) # generate some x values
p = stats.norm.pdf(x, mu, std) # calculate the y values for the normal curve
sns.lineplot(x=x, y=p, color="orange", ax=ax2)


# %% Regression analysis wind


wind = BFI_obs[['wind_speed', 'wind_direction', 'wind_gust']]

# Drop rows where all values are missing
wind = wind.dropna(how='all')

# Drop rows where wind speed AND wind gust are missing
wind = wind.dropna(subset=['wind_speed', 'wind_gust'], how='all')

# Drop rows where wind direction is missing
wind = wind.dropna(subset=['wind_direction'])

# Filter out rows that do not have wind direction NE to NW (315 to 45 degrees)
wind = wind[(wind['wind_direction'] <= 45) | (wind['wind_direction'] >= 315)]

# Filter out rows with wind speed less than 5 mph UNLESS it is accompanied by a wind gust greater than 15 mph
wind = wind[(wind['wind_speed'] > 8) | (wind['wind_gust'] > 15)]

# Create new column with max value between wind_speed and wind_gust
wind['max_wind'] = wind[['wind_speed', 'wind_gust']].max(axis=1)

# Drop wind_speed and wind_gust
wind = wind.drop(['wind_speed', 'wind_gust'], axis=1)


# Reindex to match diff
wind = wind.reindex(diff.index, method='nearest')

# Multiple linear regression for SLP and wind
# lev_top_3 = np.array([132, 100, 135])
# wind = wind.drop(wind.index[lev_top_3])
# diff = diff.drop(diff.index[lev_top_3])

x = wind['max_wind'].values.reshape(-1, 1)
y = diff.values

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
y_pred = model.predict(x)

# model = LinearRegression().fit(x, y)
# r2 = model.score(x, y)
# y_int = model.intercept_
# slope = model.coef_
# y_pred = model.predict(x)


fig, ax1 = plt.subplots(1,1)
fig.set_size_inches(14, 8)

ax1.plot(x[:,1], y, 'k.')

ax1.set_title('Predicted/Observed difference in water height compared with NW-NE wind ' + '($R^{2}$ = ' + str(round(r_sq, 4)) + ')')
ax1.set_ylabel('Water level difference (ft)')
ax1.set_xlabel('Wind speed (mph)')

ax1.plot(x[:,1], y_pred, 'r', label = '$y = ' + str(round(slope, 3)) + 'x + ' + str(round(y_int, 2)) + '$',
         linewidth=2)
ax1.legend()

# plot residuals
mu, std = stats.norm.fit(model.resid)
fig2, ax2 = plt.subplots()
sns.histplot(x=model.resid, ax=ax2, stat="density", linewidth=0, kde=True)
ax2.set(title="Distribution of residuals (wind vs storm surge)", xlabel="residual")
xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x = np.linspace(xmin, xmax, 100) # generate some x values
p = stats.norm.pdf(x, mu, std) # calculate the y values for the normal curve
sns.lineplot(x=x, y=p, color="orange", ax=ax2)


# plot residuals vs leverage
# norm_resid = model.get_influence().resid_studentized_internal
# lev = model.get_influence().hat_matrix_diag
# Cooks = model.get_influence().cooks_distance[0]

# plot_lm = plt.figure()
# plt.scatter(lev, norm_resid, alpha=0.5)
# sns.regplot(x = lev, y = norm_resid, scatter=False, ci=False, lowess=True,
#             line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
# plot_lm.axes[0].set_xlim(0, max(lev) + 0.01)
# plot_lm.axes[0].set_ylim(-3, 5)
# plot_lm.axes[0].set_title('Residuals vs Leverage')
# plot_lm.axes[0].set_xlabel('Leverage')
# plot_lm.axes[0].set_ylabel('Standardized Residuals')

# lev_top_3 = np.flip(np.argsort(Cooks), 0)[:3]
# for i in lev_top_3:
#     plot_lm.axes[0].annotate(i, xy = (lev[i], norm_resid[i]))


# %% Upstream discharge analysis

fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(14, 8)

# ax1.plot(Duwamish_obs['water_level'], label= 'E Marginal Way Bridge Water Level')
# ax1.set_title('Predicted/Observed difference in water height compared with upstream discharge.')
# ax1.set_ylabel('Water level difference (ft)')
# ax1.set_xlabel('Date')
# ax1.plot([], [], '#1f1d1a', label= 'Upstream Green River Discharge ' + '($ft^{3}$/s)')

# ax2 = ax1.twinx()
# ax2.plot(Palmer_plant['discharge'], '#1f1d1a', linewidth=3)
# ax2.set_ylabel('Discharge ($ft^{3}$/s)')
# ax1.legend()

discharge = Palmer_plant.resample('12h').max()['discharge']
discharge = discharge.reindex(diff.index, method='nearest')

ax1.plot(discharge, diff, 'k.')

# lev_top_3 = np.array([168, 71, 169])
# discharge = discharge.drop(discharge.index[lev_top_3])
# diff = diff.drop(diff.index[lev_top_3])

x = discharge.values.reshape(-1, 1)
y = diff.values

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
y_pred = model.predict(x)


# model = LinearRegression().fit(x, y)
# r2 = model.score(x, y)
# y_int = model.intercept_
# slope = model.coef_
# y_pred = model.predict(x)

ax1.plot(x[:,1], y_pred, 'r', label = '$y = ' + str(round(slope, 4)) + 'x + ' + str(round(y_int, 2)) + '$',
          linewidth=2)
ax1.set_xlabel('Discharge ($ft^{3}$/s)')
ax1.set_ylabel('Water level difference (ft)')
ax1.set_title('Predicted/Observed difference in water height compared with upstream streamflow ' + '($R^{2}$ = ' + str(round(r_sq, 2)) + ')')
ax1.legend()

# plot residuals
mu, std = stats.norm.fit(model.resid)
fig2, ax2 = plt.subplots()
sns.histplot(x=model.resid, ax=ax2, stat="density", linewidth=0, kde=True)
ax2.set(title="Distribution of residuals (discharge vs storm surge)", xlabel="residual")
xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x = np.linspace(xmin, xmax, 100) # generate some x values
p = stats.norm.pdf(x, mu, std) # calculate the y values for the normal curve
sns.lineplot(x=x, y=p, color="orange", ax=ax2)


# plot residuals vs leverage
norm_resid = model.get_influence().resid_studentized_internal
lev = model.get_influence().hat_matrix_diag
Cooks = model.get_influence().cooks_distance[0]

plot_lm = plt.figure()
plt.scatter(lev, norm_resid, alpha=0.5)
sns.regplot(x = lev, y = norm_resid, scatter=False, ci=False, lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_lm.axes[0].set_xlim(0, max(lev) + 0.01)
plot_lm.axes[0].set_ylim(-3, 5)
plot_lm.axes[0].set_title('Residuals vs Leverage')
plot_lm.axes[0].set_xlabel('Leverage')
plot_lm.axes[0].set_ylabel('Standardized Residuals')

lev_top_3 = np.flip(np.argsort(Cooks), 0)[:3]
for i in lev_top_3:
    plot_lm.axes[0].annotate(i, xy = (lev[i], norm_resid[i]))
    
    
# %% Multivariate linear regression

# New DataFrame incorporating all the independent and dependent variables to be used (size = 52704 = surge.size)
# independent variables: 
    # SLP
    # wind
    # precip
# dependent variable
    # surge

# storm surge  
surge = Elliott_Bay['surge'].dropna()
    
# sea level pressure
SLP = BFI_obs['SLP'].dropna().reindex(surge.index, method='nearest')

# wind
# wind = BFI_obs[['wind_speed', 'wind_gust', 'peak_wind_speed']].dropna(how='all')
# # Create new column with max value between wind_speed and wind_gust
# wind['max_wind'] = wind.max(axis=1, skipna=True)
# # Drop wind_speed and wind_gust
# wind_raw = wind['max_wind']
# # interpolate missing values
# int_wind = wind_raw.interpolate(method='time')
# # Reindex wind obs; interpolate nearest values to match surge index
# wind = int_wind.reindex(surge.index, method='ffill')
# wind.rename('wind', inplace=True)


# precip
# precip_6hr = BFI_obs['6hr_precip']
# # interpolate missing values
# int_precip = precip_6hr.interpolate(method='time')
# # Reindex precip obs; interpolate nearest values to match surge index
# precip = int_precip.reindex(surge.index, method='nearest').fillna(0)
# precip.rename('precip', inplace=True)
    

# New DataFrame to be used for multiple linear regression
SPU = pd.concat([surge, SLP], axis=1)

# Regression
# X = SPU[['SLP', 'wind', 'precip']]
# X = SPU[['SLP', 'wind']]
# X = SPU[['SLP', 'precip']]
# X = SPU[['wind', 'precip']]

# X = SPU[['wind']]
# X = SPU[['precip']]

# SPU_small = SPU.drop(SPU.loc[SPU.index < pd.Timestamp('2020-10-01 00:00:00', tz='UTC')].index)

X = SPU['SLP']
X = sm.add_constant(X)
y = SPU['surge']

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

# Variance Inflation Factor calculation to measure multcolinearity
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

print("\n95% confidence the true storm surge value is between " + str(PI_lower.round(3)) + "ft - " + str(PI_upper.round(3)) + "ft.\n")
print("Best guess is " + str(best_guess.round(3)) + 'ft.')
    
# Serialize regression model output with Pickle
# Create an array of realistic SLP values (800 - 1100)hPa
real_SLP = np.arange(900, 1051)
x = sm.add_constant(real_SLP)
pred = result.get_prediction(x)
frame = pred.summary_frame(alpha=0.05)

# add realistic SLP values to output DataFrame (frame)
frame['SLP_values'] = real_SLP
frame.set_index('SLP_values', inplace=True)

# save DataFrame with Pickle serialization
# filename = 'SPU_df.pkl'
# pickle.dump(frame, open(filename, 'wb'))

# write 'frame' to csv file
frame.to_csv('C:/Users/Kyle/Documents/Summer Projects 2023 - WxNet/Website Dev/marbzgrop.github.io/SPU.csv')
    
