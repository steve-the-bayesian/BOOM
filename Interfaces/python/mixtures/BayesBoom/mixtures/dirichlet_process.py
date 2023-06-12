import pandas as pd
import numpy as np

import BayesBoom.boom as boom
import BayesBoom.R as R


class DirichletProcessMvn:
    def __init__(self,
                 data,
                 concentration_parameter=1,
                 mean_base_measure: R.MvnGivenSigma = None,
                 variance_base_measure: R.WishartPrior = None):
        data = np.array(data, dtype="float")
        self._model = boom.DirichletProcessMvnModel(
            data.shape[1], float(concentration_parameter))

        # Assign data
        for i in range(data.shape[0]):
            self._model.add_data(R.to_boom_vector(data[i, :]))

        self._mean_base_measure = self.set_mean_base_measure(
            mean_base_measure, data)
        self._variance_base_measure = self.set_variance_base_measure(
            variance_base_measure, data)
        self._concentration_parameter = float(concentration_parameter)

        # Assign sampler.
        self._sampler = boom.DirichletProcessMvnCollapsedGibbsSampler(
            self._model,
            self._mean_base_measure.boom(),
            self._variance_base_measure.boom(),
            boom.GlobalRng.rng)

        self._model.set_method(self._sampler)

    @property
    def dim(self):
        return self._mean_base_measure.dim

    def mcmc(self, niter, ping=None, permute_state_labels=True, seed=None):
        """
        Run 'niter' iterations of MCMC.
        """
        niter = int(niter)
        if ping is None:
            ping = int(niter / 10)
            if ping < 0:
                ping = 1
        ping = int(ping)

        if seed is not None:
            boom.GlobalRng.rng.seed(seed)

        self._allocate_space(niter)
        for i in range(niter):
            R.print_timestamp(i, ping)
            self._model.sample_posterior()
            self._record_draws(i)

        self._gather_params()

        if permute_state_labels:
            self.choose_permutation()

        return None

    def set_mean_base_measure(self, mean_base_measure: R.MvnGivenSigma, data):
        """
        Args:
          mean_base_measure:
        """
        if mean_base_measure is None:
            ybar = data.mean(axis=0)
            mean_base_measure = R.MvnGivenSigma(ybar, 1)
        if not isinstance(mean_base_measure, R.MvnGivenSigma):
            raise Exception("mean_base_measure must either be None or an "
                            "instance of R.MvnGivenSigma.")
        return mean_base_measure

    def set_variance_base_measure(self,
                                  variance_base_measure: R.WishartPrior,
                                  data):
        """
        Args:
          variance_base_measure:  An instance of
        """
        if variance_base_measure is None:
            V = np.cov(data, rowvar=False, ddof=1)
            dim = V.shape[0]
            variance_base_measure = R.WishartPrior(dim + 1, V)
        if not isinstance(variance_base_measure, R.WishartPrior):
            raise Exception("variance_base_measure must either be None or an "
                            "instance of R.WishartPrior.")
        return variance_base_measure

    @property
    def log_likelihood(self):
        return np.array(self._log_likelihood)

    def choose_permutation(self):
        """
        Permute the model parameters and cluster labels to remove label
        switching.
        """
        for nclusters in self._params.keys():
            permutations = identify_permutation_from_labels(
                self._params[nclusters]["cluster_labels"])
            means = self._params[nclusters]["means"]
            variances = self._params[nclusters]["variances"]
            cluster_labels = self._params[nclusters]["cluster_labels"]
            niter = means.shape[0]
            for i in range(niter):
                perm = permutations[i, :]
                means[i, :, :] = means[i, perm, :]
                variances[i, :, :, :] = variances[i, perm, :, :]
                permuted_labels = np.arange(nclusters)[perm]
                cluster_labels[i] = permuted_labels[self._params[nclusters][
                    "cluster_labels"][i]]
            self._params[nclusters]["means"] = means
            self._params[nclusters]["variances"] = variances
            self._params[nclusters]["cluster_labels"] = cluster_labels

    def cluster_size_distribution(self, burn=None):
        if burn is None:
            burn = R.suggest_burn(self.log_likelihood)
        burn = int(burn)
        dist = {}
        for k in self._params.keys():
            dist[k] = np.sum(np.array(self._params[k]["iterations"]) >= burn)
        dist = pd.Series(dist)
        total = np.sum(dist)
        if total > 0:
            dist = dist / total
        return dist.sort_index()

    def cluster(self, data: np.ndarray, nclusters: int, prob: bool = False,
                burn: int = None):
        """
        Return the most likely cluster (or the cluster probabilities) of one or
        more previously unseen data points.

        Args:
          data: One or more data points.  Multiple data points are rows in a 2D
            array.
          nclusters:  The number of clusters to assume for the clustering.
          prob: If True, then the return value is a numpy array giving the
            probability of cluster 1, 2, etc.  Otherwise the most likely
            cluster for each data point is returned.
          burn: The number of MCMC iterations to remove as burn-in.  The
             default (None) indicates that

        Return:
          A numpy array.
        """

        data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape((1, -1))
        nobs = data.shape[0]

        if burn is None:
            burn = R.suggest_burn(self.log_likelihood)

        means = self._params[nclusters]["means"]
        variances = self._params[nclusters]["variances"]
        cluster_labels = self._params[nclusters]["cluster_labels"]
        if burn > 0:
            keep = self._params[nclusters]["iterations"] >= burn
            means = means[keep, :, :]
            variances = variances[keep, :, :, :]
            cluster_labels = cluster_labels[keep, :]

        niter = means.shape[0]
        if niter == 0:
            raise Exception(
                f"No clusters of size {nclusters} survived burn-in.")

        prior = np.apply_along_axis(R.table, 1, cluster_labels)
        # niter x nclusters
        total = prior.sum(axis=1).reshape((-1, 1))
        prior = prior / total

        full_cluster_prob = np.empty((niter, nobs, nclusters))

        for i in range(niter):
            logprob = np.array([np.log(prior[i, :])] * nobs)
            for k in range(nclusters):
                logprob[:, k] += R.dmvn(
                    data, means[i, k, :], variances[i, k, :, :], logscale=True)
            logprob_max = np.max(logprob, axis=1).reshape((-1, 1))
            logprob -= logprob_max
            local_prob = np.exp(logprob)
            total = local_prob.sum(axis=1).reshape((-1, 1))
            full_cluster_prob[i, :, :] = local_prob / total

        cluster_probs = full_cluster_prob.mean(axis=0)

        if prob:
            return cluster_probs
        else:
            return np.argmax(cluster_probs, axis=1)

    def _allocate_space(self, niter):
        """
        Create space for an additional 'niter' draws.
        """
        if not hasattr(self, "_params"):
            self._params = {}
        if not hasattr(self, "_log_likelihood"):
            self._log_likelihood = []

    def _ensure_params(self, number_of_clusters):
        """
        Allocate space to record a draw that has 'number_of_clusters' clusters.
        """
        if number_of_clusters not in self._params.keys():
            self._params[number_of_clusters] = {
                "means": [],
                "variances": [],
                "iterations": [],
                "cluster_labels": [],
            }

    def _record_draws(self, iteration: int):
        """
        Record the state of the current draw.

        Args:
          iteration:  The iteration number of the current draw.
        """

        number_of_clusters = self._model.number_of_clusters
        self._ensure_params(number_of_clusters)

        means = np.empty((number_of_clusters, self.dim))
        variances = np.empty((number_of_clusters, self.dim, self.dim))
        for i in range(number_of_clusters):
            means[i, :] = self._model.cluster(i).mu.to_numpy()
            variances[i, :, :] = self._model.cluster(i).Sigma.to_numpy()
        self._params[number_of_clusters]["means"].append(means)
        self._params[number_of_clusters]["variances"].append(variances)
        self._params[number_of_clusters]["iterations"].append(iteration)
        self._params[number_of_clusters]["cluster_labels"].append(
            self._model.cluster_labels)
        self._log_likelihood.append(self._model.log_likelihood)

    def _gather_params(self):
        for nclusters in self._params.keys():
            self._params[nclusters]["means"] = np.array(
                self._params[nclusters]["means"])
            self._params[nclusters]["variances"] = np.array(
                self._params[nclusters]["variances"])
            self._params[nclusters]["iterations"] = np.array(
                self._params[nclusters]["iterations"])
            self._params[nclusters]["cluster_labels"] = np.array(
                self._params[nclusters]["cluster_labels"])
        self._log_likelihood = np.array(self._log_likelihood)


def _compute_mean_cluster_probs(cluster_indicators, permutations):
    """
    Args:
      cluster_indicators: niter x nobs x nclusters array of 0/1 cluster
        indicators.  Element (i, j, k) is 1 if in MCMC iteration i observation
        j was assigned to cluster k.  The element is 0 otherwise.
      permutations: niter x nclusters array of permutations.  Each row is an
        arrangement of the nubmers 0..nclusters-1.
    """
    niter, nobs, nclusters = cluster_indicators.shape
    if permutations.shape[0] != niter:
        raise Exception("permutations has the wrong leading dimension.")
    permuted_indicators = cluster_indicators.copy()
    for i in range(niter):
        permuted_indicators[i, :, :] = cluster_indicators[
            i, :, permutations[i, :]].T

    prior = np.full((1, nobs, nclusters), 1.0 / nclusters)
    permuted_indicators = np.concatenate((permuted_indicators, prior))
    probs = permuted_indicators.mean(axis=0)
    return probs


def _solve_linear_assignment_problem(cluster_indicators: np.ndarray,
                                     mean_probs: np.ndarray):
    """
    Let nobs denote the number of observations, and nclusters the number of
    clusters.

    cluster_indicators: A [nobs x nclusters] array of 0/1 dummy variables
      indicating the cluster assigned to each observation.
    mean_probs:  The current probability

    Returns:
      permutation, total_cost
    """
    nclusters = mean_probs.shape[1]
    cost = np.empty((nclusters, nclusters))
    log_probs = np.log(mean_probs)

    for i in range(nclusters):
        for j in range(nclusters):
            # cost[i, j] is -1 times the contribution to multinomial log
            # likelihood of cluster labels == j under probability column i.
            cost[i, j] = -1 * cluster_indicators[:, i].dot(log_probs[:, j])

    lap = boom.LinearAssignment(boom.Matrix(cost))
    min_cost = lap.solve()
    permutation = lap.row_solution
    return permutation, min_cost


def identify_permutation_from_labels(cluster_labels):
    """
    Args:
      cluster_labels: A [niter x nobs] numpy array.  Element [i, j] is the
        integer cluster label for observation j in Monte Carlo iteration i.

    Returns:
      permutations:  A [niter x nclusters] numpy array of integers.
        cluster_labels[i, permutations[i, :]]
    """
    cluster_labels = np.array(cluster_labels)
    niter, nobs = cluster_labels.shape
    nclusters = int(cluster_labels.max()) + 1
    levels = np.arange(nclusters)
    encoder = R.OneHotEncoder("blah", levels=levels, baseline_level=None)

    # cluster_indicators are the dummy variable representation of
    # cluster_labels.
    cluster_indicators = np.empty((niter, nobs, encoder.dim))
    for i in range(niter):
        cluster_indicators[i, :, :] = encoder.encode(cluster_labels[i, :])

    permutations = np.array([np.arange(nclusters)] * niter).reshape(
        (niter, nclusters))

    total_cost = np.inf
    cost_reduction = np.inf
    while cost_reduction > 1e-5:
        mean_cluster_probs = _compute_mean_cluster_probs(
            cluster_indicators, permutations)

        old_total_cost = total_cost
        total_cost = 0
        for draw in range(niter):
            permutation, min_cost = _solve_linear_assignment_problem(
                cluster_indicators[draw, :, :], mean_cluster_probs)
            total_cost += min_cost
            permutations[draw, :] = permutation
        cost_reduction = old_total_cost - total_cost

    return permutations
