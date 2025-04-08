import pandas as pd
import numpy as np

import BayesBoom.boom as boom
import BayesBoom.R as R

import matplotlib.pyplot as plt

class HiddenMarkovModel:
    """
    A hidden Markov model describing the time series of events for a
    collection of subjects.  All subjects have the same parameters, but each
    subject has his own hidden Markov chain.
    """
    def __init__(self,
                 state_dim: int,
                 save_state_draws: bool = False):
        state_dim = int(state_dim)
        if state_dim <= 0:
            raise Exception("state_dim must be a positive integer")
        self._state_dim = state_dim

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

        self._log_likelihood_draws = None

        self._boom_hmm = None

        self._save_state_draws = save_state_draws
        self._state_draws = {}

    @property
    def state_dim(self):
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

    def save_state_draws(self):
        self._save_state_draws = True
        if self._boom_hmm:
            self._boom_hmm.save_state_draws()
        
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

        self._data = {subject: group["data"]
                      for subject, group in dict(tuple(grouped)).items()}
        
        # for subject, group in dict(tuple(grouped)).items():
        #     self._data[subject] = group["data"]

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

        if self._save_state_draws:
            self._boom_hmm.save_state_draws()
        
        return self._boom_hmm

    def imputed_state(self, user_id: int):
        """
        Args:
          user_id: The user index (0, 1, 2, ...)

        Returns:
          A 2d numpy array with dimensions (iteration, time), containing the
          imputed hidden Markov chain for the requested user.
        """
        if not self._save_state_draws:
            raise Exception("Imputed state values are not available.  "
                            "Please call save_state_draws() prior to "
                            "training the model.")
        return self._state_draws[user_id]

    def plot_state_distribution(self, state_distribution, burn=0,
                                time_window=None, ax=None):
        """
        
        """
        if burn < 0:
            burn = 0
        state_distribution = state_distribution[burn:, :]

        if time_window:
            state_distribution = state_distribution[:, time_window]

        niter = state_distribution.shape[0]
        time_dimension = state_distribution.shape[1]

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.set_xlim(0, time_dimension)
        ax.set_ylim(0, 1)

        counts = np.apply_along_axis(
            np.bincount,
            0,
            state_distribution,
            minlength=self.state_dim)

        # counts.shape is (state_dim, time)
        probs = counts / niter

        times = np.arange(time_dimension)
        cum_probs = np.zeros_like(probs[0, :])

        grays = np.arange(self.state_dim) / self.state_dim

        for s in range(self.state_dim):
            current_probs = probs[s, :]
            ax.bar(times,
                   current_probs,
                   width=1,
                   bottom=cum_probs,
                   color=str(grays[s]))
            cum_probs += current_probs

        return ax

    def plot_components(self, burn=0, style="ts", fig=None, ax=None, **kwargs):
        fig, ax = self._state_models[0].plot_components(
            self._state_models,
            burn=burn,
            style=style,
            fig=fig,
            ax=ax,
            **kwargs)

        return fig, ax

    def plot_loglike(self, burn=0, fig=None, ax=None, **kwargs):
        fig, ax = R.ensure_ax(fig, ax)
        if burn < 0:
            burn = 0

        niter = self._log_likelihood_draws.shape[0]
            
        iteration = range(burn, niter)
        ax.plot(iteration, self._log_likelihood_draws[burn:])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log likelihood")

        return fig, ax
    
    def _allocate_space(self, niter):
        self._log_likelihood_draws = np.empty(niter)
        self._markov_model.allocate_space(niter)
        for model in self._state_models:
            model.allocate_space(niter)

        self._state_draws = {}
        if self._save_state_draws:
            for user, data_series in self._data.items():
                self._state_draws[user] = np.empty(
                    (niter, data_series.shape[0]),
                    dtype=int)

    def _record_draw(self, iteration):
        self._log_likelihood_draws[iteration] = self._boom_hmm.loglike
        self._markov_model.record_draw(iteration)
        for model in self._state_models:
            model.record_draw(iteration)

        if self._save_state_draws:
            hidden_chain = self._boom_hmm.imputed_state
            for user, chain in enumerate(hidden_chain):
                self._state_draws[user][iteration, :] = chain

    def _assign_data_to_boom_model(self, boom_hmm, mixture_component):
        """
        Convert the data stored in this object to equivalent C++ data
        structures, and assign them to the boom_hmm model.
        """
        data_builder = mixture_component.create_boom_data_builder()
        for user, data in self._data.items():
            # The data here are built one time series at a time.
            boom_data = data_builder.build_boom_data(data)
            boom_hmm.add_data(boom_data)

    def _ensure_markov_model(self):
        if self._markov_model is None:
            self._markov_model = R.MarkovModel(state_size=self.state_dim)

        if self._markov_prior is None:
            S = self.state_dim
            prior_transition_counts = np.ones((S, S))

            if self.number_of_users > 1:
                self._markov_prior = R.MarkovConjugatePrior(prior_transition_counts)
            else:
                initial_distribution_counts = np.ones(S)
                self._markov_prior = R.MarkovConjugatePrior(
                    prior_transition_counts,
                    initial_distribution_counts)

            self._markov_model.set_prior(self._markov_prior)
