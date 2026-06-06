import BayesBoom.boom as boom
import BayesBoom.R as R
import numpy as np


class BinomialBandit:
    """
    A classroom example multi-armed bandit for success/failure outcomes.
    Each arm has an independent success probability.  Data for arms are
    communicated by counts of successes and trials.
    """
    def __init__(self, number_of_arms: int):
        self._boom_bandit = None

        self._trials = np.zeros(number_of_arms)
        self._successes = np.zeros(number_of_arms)
        
        self._prior_probs = np.repeat(.5, number_of_arms)
        self._prior_sample_sizes = np.repeat(1.0, number_of_arms)
        
    def number_of_arms(self):
        return self._trials.shape[0]

    def observe_data(self, arm, incremental_successes, incremental_trials):
        self._successes[arm] += incremental_successes
        self._trials[arm] += incremental_trials

        if (self._boom_bandit):
            self._boom_bandit.observe_data(
                int(arm),
                int(np.round(incremental_successes)),
                int(np.round(incremental_trials)))

    def update_posterior(self, ndraws):
        if not self._boom_bandit:
            self.boom()
        self._boom_bandit.update_posterior(int(ndraws))

    @property
    def optimal_arm_probabilities(self):
        if not self._boom_bandit:
            self.boom()
        return R.to_numpy(self._boom_bandit.optimal_arm_probabilities)
    
    def boom(self):
        """
        Instantiate any missing Boom objects and populate them with data
        from this class.
        """
        number_of_arms = self._prior_probs.shape[0]
        
        models = []
        for i in range(number_of_arms):
            prob = self._prior_probs[i]
            n = self._prior_sample_sizes[i]
            a = n * prob
            b = n - a
            prior = boom.BetaModel(a, b)
            model = boom.BinomialModel(prob)
            sampler = boom.BetaBinomialSampler(model, prior)
            model.set_method(sampler)
            models.append(model)

        self._boom_bandit = boom.BinomialBandit(models)
        for i in range(number_of_arms):
            self._boom_bandit.observe_data(
                int(i),
                int(np.round(self._successes[i])),
                int(np.round(self._trials[i])))

    
