import pandas as pd
import numpy as np

import BayesBoom.boom as boom
import BayesBoom.R as R


class HiddenMarkovModel:
    """
    A hidden Markov model describing the time series of events for a
    collection of subjects.  All subjects have the same parameters, but each
    subject has his own hidden Markov chain.
    """
    def __init__(self,
                 state_size: int):
        state_size = int(state_size)
        if state_size <= 0:
            raise Exception("state_size must be a positive integer")
        self._state_size = state_size

        # A dict of data sets indexed by user id.
        self._data = {}

        # A list of R "Model" objects corresponding to Boom Model objects.
        # Adding a state model to the list increases the size of the hidden
        # state space by 1.
        self._state_models = []

        # Model object representing the hidden Markov chain.
        self._markov_model = None

        # Prior distribution for self._markov_model.
        self._markov_prior = None

        # An R MarkovModel object.  This object will be created by a call to
        # "allocate_space" so that it has the right dimension.
        self._transition_model = None
        self._transition_model_prior = None

        self._log_likelihood_draws = None

        self._boom_hmm = None

    @property
    def state_size(self):
        """
        The dimension of the hidden Markov chain.
        """
        return len(self._state_models)

    @property
    def number_of_users(self):
        return len(self._data)

    @property
    def niter(self):
        """
        The number of MCMC iterations that have been run on the HMM.
        """
        if self._log_likelihood_draws is None:
            return 0
        else:
            return len(self._log_likelihood_draws)

    @property
    def markov_model(self):
        """
        The R.MarkovModel object responsible for managing the parameters of
        the hidden Markov chain.  This is where you find MCMC draws of the
        state transition probabilities.
        """
        return self._markov_model

    def add_state_model(self, model):
        """
        Args:
          state_model: An R.mixture_component object. describing the conditional
            distribution of the data for a particular state value (i.e. for a
            particular value of the unobserved Markov chain).  If the state
            model is to be trained by MCMC, it must have a prior distribution
            set, so that a posterior sampler will be created for the model when
            its 'boom' method is called.
        """
        self._state_models.append(model)


    def set_markov_prior(self, prior):
        """
        Args:
          prior: An object of class MarkovConjugatePrior serving as the prior
            distribution for the hidden Markov chain.
        """
        if not isinstance(prior, R.MarkovConjugatePrior):
            raise Exception("The prior distribution should be of class "
                            "MarkovConjugatePrior.")
        self._markov_prior = prior


    def add_data(self, data, subject_id=None, timestamp=None):
        """
        This HMM can support data from a single hidden Markov chain, or from
        multiple subjects each with their own, independent hidden Markov chain.
        If a single user model is desired, just leave subject_id as None.

        Args:
          data: A vector of observed values.  In later updates of this model we
            could extend data to be a matrix or data frame.  Each 'row' of
            'data' is the data for a particular subject at a particular time
            period.
          subject_id: A vector of strings or other identifiers of the same
            length as 'data'.  Each element identifies the subject associated
            with the corresponding row of 'data'.  If None then the model
            assumes there is a single subject.
          timestamp: A numeric vector of othe same lenghth as 'data', or None.
            If None then each element of 'data' is added to the corresponding
            subject in the order observed.  Otherwise
        """
        if timestamp is None:
            timestamp = range(len(data))

        if subject_id is None:
            subject_id = np.zeros(len(data), dtype=int)

        # split thed data into groups by subject and sort by timestamp
        frame = pd.DataFrame(
            {
                "data": data,
                "subject_id": subject_id,
                "timestamp": timestamp
            }
        )

        frame = frame.sort_values(by=["subject_id", "timestamp"])
        grouped = frame.groupby("subject_id")
        # import pdb
        # pdb.set_trace()
        self._data = {
            subject: group["data"]
            for subject, group in dict(tuple(grouped)).items()
        }


    def train(self, niter, ping=100):
        """
        Sample the posterior distribution of the HMM using the forward
        backward posterior sampler described in Scott (2002).

        Args:
          niter:  The number of MCMC iterations to run.
          ping: The number of iterations in between status updates printed to
            the screen.  If status updates are not desired set 'ping' to a
            negative number, or to None.

        Effects:
          - The log likelihood of each draw is recorded in
            self._log_likelihood_draws.
          - The value of the hidden Markov transition
            probability matrix is stored in self._transition_matrix_draws.
          - The MCMC histories for each user are stored in
            self._user_draws[user_id]
          - The draws for the parameters of state model i are stored in
            self._state_models[i]
        """
        self.boom()
        self._allocate_space(niter)

        for i in range(niter):
            R.print_timestamp(i, ping)
            self._boom_hmm.sample_posterior()
            self._record_draw(i)


    def boom(self):
        """
        Create (if necessary) and return the BOOM HiddenMarkovModel object.
        Subsequent
        """
        if self._boom_hmm is not None:
            return self._boom_hmm

        boom_components = [model.boom() for model in self._state_models]
        boom_component_list = boom.MixtureComponentVector()
        for model in boom_components:
            boom_component_list.append(model)

        self._ensure_markov_model()

        self._boom_hmm = boom.HiddenMarkovModel(
            boom_component_list.values,
            self._markov_model.boom())

        self._boom_hmm_sampler = boom.HmmPosteriorSampler(self._boom_hmm,
                                                          boom.GlobalRng.rng)
        self._boom_hmm.set_method(self._boom_hmm_sampler)

        self._assign_data_to_boom_model(self._boom_hmm, self._state_models[0])

        return self._boom_hmm

    def _allocate_space(self, niter):
        self._log_likelihood_draws = np.empty(niter)
        self._markov_model.allocate_space(niter)
        for model in self._state_models:
            model.allocate_space(niter)

    def _record_draw(self, iteration):
        self._log_likelihood_draws[iteration] = self._boom_hmm.loglike
        self._markov_model.record_draw(iteration)
        for model in self._state_models:
            model.record_draw(iteration)

    def _assign_data_to_boom_model(self, boom_hmm, mixture_component):
        """
        Convert the data stored in this object to equivalent C++ data
        structures, and assign them to the boom_hmm model.
        """
        data_builder = mixture_component.create_boom_data_builder()
        for user, data in self._data.items():
            boom_data = data_builder.build_boom_data(data)
            boom_hmm.add_data(boom_data)



    def _ensure_markov_model(self):
        if self._markov_model is None:
            self._markov_model = R.MarkovModel(state_size=self.state_size)

        if self._markov_prior is None:
            S = self.state_size
            prior_transition_counts = np.ones((S, S))

            if self.number_of_users > 1:
                self._markov_prior = R.MarkovConjugatePrior(prior_transition_counts)
            else:
                initial_distribution_counts = np.ones(S)
                self._markov_prior = R.MarkovConjugatePrior(
                    prior_transition_counts,
                    initial_distribution_counts)

            self._markov_model.set_prior(self._markov_prior)
