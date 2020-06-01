from abc import ABC, abstractmethod


# ===========================================================================
class StateModel(ABC):
    """StateModel objects are wrappers around boom.StateModel objects, which are
    opaquie C++ types.  The primary purpose of a separate python object is to
    handle various accounting duties required of StateModel's by the Bsts model
    defined below.

    """

    def __init__(self):
        """
        Args:
          state_index_begin: The index of the state subcomponent described by
            this state model, in the full state vector.

        """
        self._state_index = -1

    @property
    def state_index(self):
        """
        The index of the this state model's first element, in the global state
        vector.  This starts off as -1.  When the state model is added to a
        bsts model, bsts will call set_state_index() to inform the state model
        of this value.
        """
        if self._state_index < 0:
            raise Exception("Each state model must be told where its state"
                            "component begins in the global state vector.  "
                            "Try calling set_state_index.")
        return self._state_index

    def set_state_index(self, state_index_begin: int):
        """
        Inform the state model of the first index in the larger state vector
        corresponding to the state component it is responsible for.

        Args:
           state_index_begin: The smallest index in the global state vector
             corresponding to the component of state that this model describes.
        """
        self._state_index = state_index_begin

    @property
    @abstractmethod
    def state_dimension(self):
        """
        The dimension of the state subcomponent managed by this model.
        """

    @abstractmethod
    def allocate_space(self, niter, time_dimension):
        """
        Allocate the space needed to call 'record_state' the given number of
        times.

        Args:
          niter:  Number of iterations (draws) to be stored.
          time_dimension:  Number of time points in the training data.
        """

    @abstractmethod
    def record_state(self, iteration, state_matrix):
        """
        Record the state of any model parameters, and the subset of the state
        vector associated with this model, so they can be analyzed later.

        Args:
          iteration:  The (integer) index of the MCMC iteration to be recorded.
          state_matrix: The current matrix containing state.  Each column is a
            state vector associated with the corresponding time point.
        """

    @abstractmethod
    def plot_state_contribution(self, ax, **kwargs):
        """
        Plot the contribution of this state model to the conditional mean at
        time t on the supplied set of axes.  This is often a dynamic
        distribution plot.
        """
