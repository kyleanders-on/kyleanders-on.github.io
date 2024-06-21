import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy import stats


# Regression analysis and data visualization
# Perform Ordinary Least Squares (OLS) linear regression and fit the model


def reg_model(SLP_data, elliott_bay_obs, tide_pred):
    """
    Create linear regression model object.
    Independent variable X: Sea level pressure.
    Dependent variable y: Storm surge (difference between water level observations and tide predictions).

    Parameters:
        SLP_data (pandas.DataFrame): Airport SLP observations Oct 1-April 1.
        elliott_bay_obs (pandas.DataFrame): Elliott Bay water level observations Oct 1-April 1.
        tide_pred (): Elliott Bay base tide predictions from NOAA Oct 1-April 1.

    Returns:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): Linear regression model object.
        X (pandas.DataFrame): Independent variable (SLP) and intercept column.
        y (pandas.Series): Dependent variable (storm surge).
    """
    SLP = SLP_data["SLP"].dropna()
    surge = (elliott_bay_obs - tide_pred).dropna().reindex(SLP.index, method="nearest")

    X = SLP
    X = sm.add_constant(X)
    y = surge["water_level"]
    model = sm.OLS(y, X).fit()

    # Print summary statistics
    print(model.summary())
    return model, X, y


# plot best fit line
def best_fit_plot(model, X, y, wx_station_code, tidal_station_name):
    reg_line = model.predict(X)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    ax.plot(X["SLP"].values, y.values, "k.")
    ax.plot(X["SLP"].values, reg_line.values, "r", label="OLS model fit")
    ax.set_xlabel(f"{wx_station_code} Sea Level Pressure (hPa)")
    ax.set_ylabel(f"{tidal_station_name} Storm surge estimate (ft)")
    ax.set_title(
        f"Observed water level/base tide prediction difference vs sea level pressure ($R^2$ = {round(model.rsquared, 2)})"
    )


# Compare residuals distribution to a normal curve
def resid_distribution(model):
    mu, std = stats.norm.fit(model.resid)
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(10, 7)
    sns.histplot(x=model.resid, ax=ax2, stat="density", linewidth=0, kde=False)
    sns.kdeplot(
        x=model.resid,
        ax=ax2,
        label="Kernel Density Estimation",
        linewidth=2.5,
        color="g",
    )
    ax2.set(title="Distribution of residuals (SLP vs storm surge)", xlabel="residual")
    xmin, xmax = plt.xlim()  # the maximum x values from the histogram above
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    sns.lineplot(
        x=x, y=p, color="orange", ax=ax2, label="Normal Distribution", linewidth=2.5
    )


# plot residuals vs leverage
def resid_leverage_comparison(model):
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
def model_prediction_output(model, mean_difference):
    """
    tete

    Parameters:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): Linear regression model object.
        mean_difference (float): mean difference between Elliott Bay and Duwamish Waterway water level observations.

    Returns:
        model_output (pandas.DataFrame): Regression model mean prediction and intervals.
    """

    # Create an array of realistic SLP values (900-1050 hPa)
    real_SLP = np.arange(900, 1051)
    x = sm.add_constant(real_SLP)
    pred = model.get_prediction(x)
    model_output = pred.summary_frame(alpha=0.05)

    # Add realistic SLP values to output DataFrame
    model_output["SLP_values"] = real_SLP
    model_output.set_index("SLP_values", inplace=True)
    print(model_output)

    # adjust model_output to account for differences between Elliott Bay
    # and Duwamish Waterway water levels
    model_output.loc[:, ["mean", "obs_ci_lower", "obs_ci_upper"]] += mean_difference

    # Approximate adjustment for mean water level difference between
    # Elliott Bay and the Duwamish Waterway near South Park
    # model_output['mean', 'upper', 'lower'] = ... + 0.3

    # Plot prediction intervals
    # ax.plot(model_output["obs_ci_lower"], "b--")
    # ax.plot(model_output["obs_ci_upper"], "b--", label="95% prediction interval")
    # ax.legend()
    # plt.show()

    # Write model_output to csv file
    # formatted_station_name = tidal_station_name.replace(" ", "_")
    # model_output.to_csv(f"./{formatted_station_name}.csv")
    return model_output
