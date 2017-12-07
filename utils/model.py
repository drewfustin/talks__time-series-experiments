import pandas as pd
import pymc3 as pm
from theano import scan, shared
import theano.tensor as tt


def build_model(X, treatment_start, treatment_observations):
    time_seen = pd.to_datetime(treatment_start) + pd.DateOffset(treatment_observations - 1)
    y = shared(X[:time_seen].values)
    y_switch = shared(X[:time_seen].index < treatment_start)
    with pm.Model() as i1ma1:
        σ = pm.HalfCauchy('σ', beta=2.)
        θ = pm.Normal('θ', 0., sd=2.)
        β = pm.Normal('β', 0., sd=2.)

        y_adj = tt.switch(y_switch, y, y - tt.dot(y, β))

        # ARIMA (0, 1, 1)
        # ---------------
        # (1 - B) y[t] = (1 - θB) ε[t]
        # y[t] - y[t-1] = ε[t] - θ * ε[t-1]
        # ε[t] = y[t] - y[t-1] - θ * ε[t-1]
        def calc_next(y_lag1, y_lag0, ε, θ):
            return y_lag0 - y_lag1 - θ * ε

        # Initial noise guess -- let's just seed with 0
        ε0 = tt.zeros_like(y_adj)

        ε, _ = scan(fn=calc_next,
                    sequences=dict(input=y_adj, taps=[-1, 0]),
                    outputs_info=[ε0],
                    non_sequences=[θ])

        pm.Potential('like', pm.Normal.dist(0, sd=σ).logp(ε))
    return i1ma1
