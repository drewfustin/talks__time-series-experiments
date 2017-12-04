import pandas as pd
import numpy as np


def hpp(λ, n, t0, freq=None, verbose=False):
    """
    Homogeneous Poisson Point Process

    Parameters
    ----------
    λ : Poisson rate parameter [transactions per day]
    n : number of transactions to produce
    t0 : initial datetime for simulation
    freq : Pandas resample rule for returned time series

    Returns
    -------
    Time series of transactions per freq
    """
    # Times between Poisson-distributed events are exponentially distributed with mean 1 / λ
    transaction_times = pd.TimedeltaIndex(np.cumsum(np.random.exponential(1 / λ, n)), unit='d')
    # Generate non-grouped HPP
    transactions = pd.Series(data=1,
                             index=pd.to_datetime(t0) + transaction_times,
                             name='transactions')
    # Check mean and variance to ensure this is Poisson-like
    if verbose:
        transactions_check = transactions.resample('1D').sum().fillna(0)[:-1]
        print('Transaction times :')
        print(transaction_times)
        print('Mean : {}'.format(transactions_check.mean()))
        print('Variance : {}'.format(transactions_check.var()))
        print()
    # Group transactions by freq
    if freq:
        transactions = transactions.resample(freq).sum().fillna(0)[1:-1]

    return transactions


def nhpp(λ, n, t0, freq=None, thinning_components=None):
    """
    Nonhomogeneous Poisson Point Process

    Parameters
    ----------
    λ : Poisson rate parameter [transactions per day] of the bounding Homogeneous Point Process
        Equal to the instantaneous rate of the Nonhomogeneous Point Process when thinning = 1
    n : number of transactions to produce
    t0 : initial datetime for simulation
    freq : Pandas resample rule for returned time series
    thinning_components : components to pass to thinning function

    Returns
    -------
    Time series of transactions per freq
    """
    # Get a non-grouped HPP
    transactions = hpp(λ, n, t0)
    # Thin the HPP
    if thinning_components:
        transactions = transactions[np.random.uniform(size=len(transactions)) <
                                    thinning(t=transactions.index, components=thinning_components)]
    # Group transactions by freq
    if freq:
        transactions = transactions.resample(freq).sum().fillna(0)[1:-1]

    return transactions


def thinning(t, components):
    """
    Thinning function used to accept or reject Homogeneous Poisson Point Process events.
    Functions in components are combined multiplicatively and must be strictly positive.
    Resulting composite is scaled to be bound by [0, 1]
        0 --> instantaneous Poisson rate = 0 in the Nonhomogeneous Process
        1 --> instantaneous Poisson rate = rate of Homogeneous Process in Nonhomogeneous Process

    Parameters
    ----------
    t : interable containing datetimes for the time series
    components : list of dicts with entries:
                     'function': function to be applied  [required]
                     'params': dict with required parameters for function  [optional]
                 e.g. components = [
                     {'function': func1, 'args': (1, 'foo')},
                     {'function': func2, 'kwargs': {'p3': 'bar', 'p4': None}}
                     ]
                 where
                 def func1(t, p1, p2):  <-- p1=1, p2='foo'
                     ...
                 and
                 def func2(t, p3, p4):  <-- p3='bar', p4=None
                     ...

    Returns
    _______
    Pandas series with index t and values [0, 1] corresponding to thinning to be applied.
    """
    y = pd.Series(1, index=t, name='thinning')
    for comp in components:
        y *= comp['function'](t, comp['params'])
    return y


def _trend(t, params):
    """
    Trend component for time series decomposition
    Modeled as exponential growth relative to unity at some t_min

    params : {
        'percent_increase' : number between [0, 1] quantifying growth rate
        'period' : the time (in days) over which percent_increase applies
        }
    """
    relative_time = (t - t.round('D').min()) / np.timedelta64(1, 'D')
    y = np.power(1 + params['percent_increase'], relative_time / params['period'])
    return y / max(y)


def _cyclical(t, params):
    """
    Cyclical component for time series decomposition
    Modeled as a fractional damping applied to a given day of week

    params : {
        'weekday_factor': list (len=7) mapping weekday to reduction between [0, 1] applied to rate
                           value corresponding to Monday is in 0-index of weekday_factor
        }
    """
    y = pd.Index(t.weekday.to_series().map({k: params['weekday_factor'][k] for k in range(7)}))
    return y / max(y)


def _seasonal(t, params):
    """
    Seasonal component for time series decomposition
    Modeled as a sine with 1-day period, amplitude between [0, 1], constrained to unity at peak

    params : {
        'peak_time' : time (str or pd.datetime) at which _seasonal is defined to be 1
        'amplitude' : number between [0, 1] dictating the percent reduction at min
        }
    """
    relative_time = (t - pd.to_datetime(params['peak_time'])).seconds / (24 * 60 * 60)
    y = 1 - params['amplitude'] * (1 + np.sin(2 * np.pi * relative_time)) / 2
    return y / max(y)
