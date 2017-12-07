from scipy.stats import beta
import numpy as np


def conversion_test(a, b, prior=None, cl=0.95, n_sample=100000):
    """
    Compare conversion rates in a vs. b and return the cl confidence band on the difference

    Parameters
    ----------
    a : iterable (population, successes)
        population : int
            number of total observations in population
        successes : int
            number of successes in population, where conversion rate = conversions / population
    b : iterable (population, successes)
        Same as a, but for the variant population in the experiment
    prior : iterable (population, successes)
        The Bayesian prior updated with observations a, b
        If None, use an uninformed flat prior
    cl : float (0, 1)
        The desired confidence band size for the returned difference
    n_sample : int
        Number of samples to draw from each distribution in determining the effect size
        Larger values of n_sample give a more precise measure of the effect size

    Returns
    -------
    (effect_size_expectation, effect_size_lower, effect_size_upper) : (float, float, float)
        Expectation value and lower and upper bounds of the effect size of the experimental variant
            between the two populations. Positive values mean b experiences a lift above a.
    """
    if not prior:
        prior = (0, 0)
    a_dist = beta(a[1] + prior[1] + 1, a[0] - a[1] + prior[0] - prior[1] + 1)
    b_dist = beta(b[1] + prior[1] + 1, b[0] - b[1] + prior[0] - prior[1] + 1)
    effect_size = b_dist.rvs(n_sample) - a_dist.rvs(n_sample)
    return (effect_size.mean(),
            np.percentile(effect_size, (1 - cl) / 2 * 100),
            np.percentile(effect_size, (1 - (1 - cl) / 2) * 100))
