from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class DoubleModel(ABC):
    """
    Base class for models that can evaluate log-probability of a real-valued
    scalar (i.e. boom.DoubleModel).
    """

    @abstractmethod
    def boom(self):
        """Return the corresponding boom.DoubleModel."""

    @property
    @abstractmethod
    def mean(self):
        """The mean of the distribution."""

    def create_boom_data_builder(self, data=None):
        from BayesBoom.R.boom_data_builders import DoubleDataBuilder
        return DoubleDataBuilder()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class MixtureComponent(ABC):
    """Base class for models that can serve as mixture components."""

    @abstractmethod
    def allocate_space(self, niter):
        """Allocate storage for 'niter' MCMC draws of the model parameters."""

    @abstractmethod
    def record_draw(self, iteration):
        """Record the current model parameters at the given MCMC iteration."""

    @abstractmethod
    def create_boom_data_builder(self, data=None):
        """
        Return a DataBuilder that converts Python data to the boom data type
        expected by this model.
        """

    @abstractmethod
    def plot_components(self, components, burn, style: str, fig, ax, **kwargs):
        """
        Plot MCMC draws for a collection of mixture components of this type.
        """


