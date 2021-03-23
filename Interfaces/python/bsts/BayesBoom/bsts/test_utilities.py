import numpy as np
import scipy.stats as stats


def simulate_random_walk(nsteps, sd=0.1):
    return np.cumsum(np.random.randn(nsteps) * sd)


def simulate_student_random_walk(nsteps, sd=0.1, nu=3):
    errors = stats.t.rvs(df=nu, scale=sd, size=nsteps)
    return np.cumsum(errors)


def simulate_local_linear_trend(nsteps, level_sd=0.1, slope_sd=0.05, mu0=0):
    slope = simulate_random_walk(nsteps, sd=slope_sd)
    error = np.random.randn(nsteps) * level_sd

    slopes = np.array([0.0] + slope[:-1].tolist())
    errors = np.array([0.0] + error[:-1].tolist())
    trend = mu0 + np.cumsum(slopes) + np.cumsum(errors)
    return trend


def simulate_student_local_linear_trend(nsteps, level_sd=0.1, level_df=3,
                                        slope_sd=0.05, slope_df=5, mu0=0):
    slope = simulate_student_random_walk(nsteps, sd=slope_sd, nu=slope_df)
    error = stats.t.rvs(size=nsteps, scale=level_sd, df=level_df)

    slopes = np.array([0.0] + slope[:-1].tolist())
    errors = np.array([0.0] + error[:-1].tolist())
    trend = mu0 + np.cumsum(slopes) + np.cumsum(errors)
    return trend
