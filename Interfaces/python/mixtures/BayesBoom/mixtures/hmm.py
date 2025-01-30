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

        # A list of R "Model" objects corresponding to Boom Model objects.
        # Adding a state model to the list increases the size of the hidden
        # state space by 1.
        self._state_models = []

        self._markov_model = None

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
        
    def add_state_model(self, model):
        """
        A 'state model' is the conditional distribution of the observed data
        given value of the the hidden Markov chain.  State models are added in
        order.  If a state model is to be trained, then it must have a posterior
        sampler set, and it must be able to consume the data passed to the HMM
        in a call to 'add_data'.
        """
        self._state_models.append(model)


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
            R.print_timestamp(i, niter)
            self._boom_hmm.sample_posterior()
            self._record_draw(i)


    def boom(self):
        """
        Create (if necessary) and return the BOOM HiddenMarkovModel object.
        Subsequent
        """
        if self._boom_hmm is not None:
            return self._boom_hmm

        components = [model.boom() for model in self._state_models]
        self._ensure_markov_model()
        
        self._boom_hmm = boom.HiddenMarkovModel(
            components, self._markov_model.boom())

        self._boom_hmm_sampler = boom.HmmPosteriorSampler(self._boom_hmm,
                                                          boom.GlobalRng.rng)
        self._boom_hmm.set_method(self._boom_hmm_sampler)

        return self._boom_hmm
            
    def _allocate_space(self, niter):
        self.finalize_model_structure();
        S = self.state_size
        self._transition_matrix_draws = np.empty((niter, S, S))
        for model in self._state_models:
            model.allocate_space(niter)
        
    def _record_draw(self, iteration):
        self._transition_matrix_draws[iteration, :, :] = (
            self._boom_hmm.markov_model.transition_matrix.to_numpy()
        )

        for model in self._state_models:
            model.record_draw(i)

    def _ensure_markov_model(self):
        if self._markov_model is None:
            self._markov_model = R.MarkovModel(self.state_size)

            S = self.state_size
            prior_transition_counts = np.ones((S, S)) 
            
            if self.number_of_users > 1:
                prior = R.MarkovConjugatePrior(prior_transition_counts)
            else:
                initial_distribution_counts = np.ones(S) 
                prior = R.MarkovConjugatePrior(prior_transition_counts,
                                               initial_distribution_counts)
                
            self._markov_model.set_prior(prior)
                
            
