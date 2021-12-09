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

    def mcmc(self, niter, ping=None, seed=None):
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

#        self._collect_draws()
        return None

    def _allocate_space(self, niter):
        """
        Create space for an additional 'niter' draws.
        """
        if not hasattr(self, "_params"):
            self._params = {}
        if not hasattr(self, "_log_likelihood"):
            self._log_likelihood = []

    def _ensure_params(self, number_of_clusters):
        if number_of_clusters not in self._params.keys():
            self._params[number_of_clusters] = {
                "means": [],
                "variances": [],
                "iterations": []
            }

    def _record_draws(self, iteration):
        """
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
        self._log_likelihood.append(self._model.log_likelihood)

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
