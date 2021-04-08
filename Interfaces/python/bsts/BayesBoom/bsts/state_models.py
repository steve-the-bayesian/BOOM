from abc import ABC, abstractmethod
import BayesBoom.R as R
import numpy as np


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
    def label(self):
        """
        A string indicating how this state model should be labelled, e.g. when
        assessing contributions of mutliple state components.
        """

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
    def record_state(self, iteration: int, state_matrix: np.ndarray):
        """
        Record the state of any model parameters, and the subset of the state
        vector associated with this model, so they can be analyzed later.

        Args:
          iteration:  The (integer) index of the MCMC iteration to be recorded.
          state_matrix: The current matrix containing state.  Each column is a
            state vector associated with the corresponding time point.

        Effect:
          The current state of the model parameters is stored.
        """

    @abstractmethod
    def restore_state(self, iteration: int):
        """
        Restore the state of the managed boom state model to the requested
        iteration.
        """

    @property
    @abstractmethod
    def state_contribution(self):
        """
        A niter x time_dimension array giving the contribution of this state
        component to the overall mean of y at each point in time.
        """

    def plot_state_contribution(
            self, fig, gridspec, time, burn=None, ylim=None, **kwargs):
        self.plot_state_contribution_default(
            fig=fig, gridspec=gridspec, time=time, burn=burn, ylim=ylim,
            **kwargs)

    def plot_state_contribution_default(
            self, fig, gridspec, time, burn=None, ylim=None, **kwargs):
        """
        A default implementation of plot_state_distribution.

        Plots the contribution of this state model to the conditional mean at
        time t on the supplied set of axes using a dynamic distribution plot.

        Args:
          ax:  The axes object on which to draw the plot.
          time:  The time axis.
          burn:  The number of iterations to discard as burn-in.
          ylim:  A pair giving the lower and upper limits on the y axis.
          kwargs:  Extra arguments passed to plot_dynamic_distribution.

        Effects:
          Creates a dynamic distribution plot on the supplied set of axes.

        Returns:
          The axes object.
        """
        if burn > 0:
            curves = self._state_contribution[int(burn):, :]
        else:
            curves = self._state_contribution

        ax = fig.add_subplot(gridspec)

        R.plot_dynamic_distribution(
            curves=curves,
            timestamps=time,
            ax=ax,
            ylim=ylim,
            **kwargs)

        return ax

    def observe_time_dimension(self, time_dimension):
        """
        Args:
          time_dimension: The number of time points being modeled.  For
            training, this is the length of the training data.  For prediction
            it is the length of the training data plus the forecast horizon.
        """
        self._state_model.observe_time_dimension(time_dimension)
