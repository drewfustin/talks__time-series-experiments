import pandas as pd
import pymc3 as pm
import theano


def build_model(X, treatment_start, treatment_observations):
    time_seen = pd.to_datetime(treatment_start) + pd.DateOffset(treatment_observations - 1)
    y = theano.shared(X[:time_seen].values)
    y_switch = theano.shared(X[:time_seen].index < treatment_start)
    with pm.Model() as i1ma1:
        σ = pm.HalfCauchy('σ', beta=2.)
        θ = pm.Normal('θ', 0., sd=2.)
        ω = pm.Normal('ω', 0., sd=5.)
        δ = pm.Normal('δ', 0, sd=10.)

        y_adj = theano.tensor.switch(y_switch, y, y - theano.tensor.dot(y, ω))

        # ARIMA (0, 1, 1)
        # ---------------
        # (1 - B) y[t] = (1 - θB) ε[t] + δ
        # y[t] - y[t-1] = ε[t] - θ * ε[t-1] + δ
        # ε[t] = y[t] - y[t-1] - θ * ε[t-1] - δ
        def calc_next(y_lag1, y_lag0, ε, θ, δ):
            return y_lag0 - y_lag1 - θ * ε - δ

        # Initial noise guess -- let's just seed with 0
        ε0 = y_adj[0] - 2 * δ

        ε, _ = theano.scan(fn=calc_next,
                           sequences=dict(input=y_adj, taps=[-1, 0]),
                           outputs_info=[ε0],
                           non_sequences=[θ, δ])

        pm.Potential('like', pm.Normal.dist(0, sd=σ).logp(ε))
    return i1ma1
