from typing import Union, Tuple, Sequence, Callable
from datetime import datetime, date
import numpy as np
import pymc3 as pm
from theano import scan, shared
import theano
import theano.tensor as tt


class TimeSeriesModel:
    def __init__(self, model: Callable) -> None:
        self.model = model

    def fit(self, ts: Sequence[float], draws: int = 2000, **kwargs) -> 'TimeSeriesModel':
        with self.model(ts, **kwargs):
            self.trace_ = pm.sample(draws)
        return self

    def predict(self, ts):
        raise NotImplementedError


def sarima_interrupted(ts: Sequence[float],
                       interruption: int,
                       order: Tuple[int, int, int]=(0, 0, 0),
                       seasonal_order: Tuple[int, int, int, int]=(0, 0, 0, 0),
                       trend: bool=False) -> pm.Model:
    """
    Seasonal ARIMA time series model to estimate interruption effect size ω

    Parameters
    ----------
    ts : sequence of floats
        The time series to be fit, assumed to be fixed period between values
    interruption : int
        Effect size ω applies to all ts for which index >= interruption
    order : (int, int, int), optional -> (p, d, q)
        ARIMA order, AR(p) I(d) MA(q)
    seasonal_order : (int, int, int, int), optional -> (P, D, Q, s)
        Seasonal ARIMA order, AR(P) I(D) MA(Q) with period s timesteps
    trend : bool, optional
        Whether to add a constant term δ to the model given drift δ/(1 − Σφi)

    Returns
    -------
    PyMC3 model with traces for each required parameter in the model.
    σ : root-variance of the white noise terms ε of the model
    ω : effect size for which X_t = Y_t * (1 + ω) for all t >= interruption
         Note: X_t are the raw observations, whereas Y_t is what the time
               series would be would with no interruption, where a single
               ARIMA model would be fit across the interruption boundary
    δ : intercept of the ARIMA model, giving rise to drift δ/(1 − Σφi)
    φ_1, φ_2, ..., φ_p : p parameters describing the autoregressive model
    θ_1, θ_2, ..., θ_q : q parameters describing the moving average model
    Φ_1, Φ_2, ..., Φ_P : P parameters describing the autoregressive model
                           with seasonal period s
    Θ_1, Θ_2, ..., Θ_Q : Q parameters describing the moving average model
                           with seasonal period s
    """
    X = tt.vector('X')
    p = tt.iscalar('p')
    d = tt.iscalar('d')
    q = tt.iscalar('q')
    P = tt.iscalar('P')
    D = tt.iscalar('D')
    Q = tt.iscalar('Q')
    s = tt.iscalar('s')
    with pm.Model() as model:
        pass
    return model


