import numpy as np


def covers(draws, truth, confidence):
    """
    Private function used to implement check_mcmc_vector and check_mcmc_matrix.
    Returns True iff a centeral 'confidence' interval constructed from 'draws'
    contains 'truth'.

    Args:
      draws:  A numpy vector containing Monte Carlo draws of a quantity.
      truth:  The target value that 'draws' are trying to cover.
     confidence: The probability content of the credibility interval used to
       check coverage.
    """
    sorted_draws = np.sort(np.array(draws).flatten())
    lower = .5 * (1 - confidence)
    upper = 1 - lower
    n = sorted_draws.shape[0]
    lo = sorted_draws[int(np.round(n * lower))]
    hi = sorted_draws[int(np.round(n * upper))]
    return truth >= lo and truth <= hi


def check_mcmc_vector(draws, truth, confidence=0.95):
    """
    Check to see if a vector of Monte Carlo draws covers a known value.

   Args:
     draws:  The vector of Monte Carlo draws to check.
     truth:  The true value against which 'draws' will be checked.
     confidence: The probability content of the credibility interval used to
       check coverage.

   Returns:

     A boolean.  True indicates that a central credibility interval with
     probability content 'confidence', constructed from 'draws', covers
     'truth'.  A False value indicates that the interval does not cover.
    """
    return covers(draws, truth, confidence)


def check_mcmc_matrix(draws, truth, confidence: float = .95,
                      control_multiple_comparisons: bool = True):
    """
    Args:
      draws: numpy matrix of draws.  Each row is a draw.
      truth: numpy vector of true values, with length matching the number of
        columns in draws.
     confidence: The probability content of the credibility interval used to
       check coverage.
     control_multiple_comparisons: If True, then the fraction of intervals
       successfully covering the true value can be slightly less than
       'confidence' as long as it is consistent with the null hypothesis of
       'confidence'-level coverage, in the sense that it is no less than two
       binomial standard errors below 'confidence'.
    """
    if confidence <= 0 or confidence >= 1:
        raise Exception("Confidence must be between 0 and 1.")
    if confidence < .5:
        confidence = 1 - confidence

    draws = np.array(draws)
    if len(draws.shape) != 2:
        raise Exception("draws must be a matrix")

    dim = draws.shape[1]
    if len(truth) != dim:
        raise Exception("The vector of true values must match the "
                        "number of columns in draws.")

    coverage_indicators = np.full(dim, False)
    fails_to_cover = 0
    for i in range(dim):
        coverage_indicators[i] = covers(draws[:, i], truth[i], confidence)
        if not coverage_indicators[i]:
            fails_to_cover += 1

    fraction_failing_to_cover = fails_to_cover / dim
    coverage_rate_limit = confidence
    if control_multiple_comparisons:
        se = np.sqrt(confidence * (1 - confidence) / dim)
        coverage_rate_limit -= 2 * se
    failure_rate_limit = 1 - coverage_rate_limit
    return fraction_failing_to_cover < failure_rate_limit


def check_stochastic_process(draws: np.ndarray,
                             truth: np.ndarray,
                             confidence: float = .95,
                             sd_ratio_threshold: float = .1,
                             control_multiple_comparisons: bool = True):
    """
    Args:
      draws: A matrix of Monte Carlo draws to be checked.  Each row is a draw
        and each column is a variable.
      truth: A vector of true values against which draws will be compared.
        truth.size() must match ncol(draws).
      confidence: The confidence associated with the marginal posterior
        intervals used to determine coverage.
      sd_ratio_threshold: One of the testing diagnostics compares the standard
        deviation of the centered draws to the standard deviation of the true
        function.  If that ratio is less than this threshold the diagnostic is
        passed.

    Returns:
      A string containing an error message describing the mode of the failure
      to cover.
    """
    import BayesBoom.boom as boom
    return boom.check_stochastic_process(
        boom.Matrix(draws),
        boom.Vector(truth),
        float(confidence),
        float(sd_ratio_threshold),
        bool(control_multiple_comparisons))
