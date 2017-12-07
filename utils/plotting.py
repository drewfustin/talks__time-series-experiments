import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import seaborn as sns


def tsplot(ts, lags=None):
    """
    Plot the time series with ACF and PACF
        Stolen from Tom Augspurger's fantastic post on pandas and time series forecasting
        https://tomaugspurger.github.io/modern-7-timeseries
    """
    fig = plt.figure()
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    ts.plot(ax=ts_ax)
    smt.graphics.plot_acf(ts, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(ts, lags=lags, ax=pacf_ax)
    [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax
