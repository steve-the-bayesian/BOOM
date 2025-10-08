import pandas as pd
import numpy as np

import BayesBoom.boom as boom
import BayesBoom.R as R

import matplotlib.pyplot as plt


class FiniteMixtureModel:
    """
    """

    def __init__(self):
        self._mixing_distribution = None
        self._mixture_components = []

        self._boom_model = None

        self._class_probs = None


    def add_component(self, component: R.MixtureComponent):
        self._mixture_components.append(component)

    @property
    def num_components(self):
        return len(self._mixture_components)

    @property
    def sample_size(self):
        """
        The number of rows in the training data.
        """
        if hasattr(self._data, "shape"):
            return self._data.shape[0]
        else:
            return len(self._data)

    def add_data(self, data):
        """
        Add data to the model.

        """
        self._data = data

    def train(self, niter, ping=None):
        """
        """
        self.boom()
        self._allocate_space(niter)

        for i in range(niter):
            R.print_timestamp(i, ping)
            self._boom_model.sample_posterior()
            self._record_draw(i)

    def boom(self):
        if self._boom_model is None:
            boom_components = [component.boom() for component in self._mixture_components]
            boom_component_list = boom.MixtureComponentVector()
            for model in boom_components:
                boom_component_list.append(model)

            self._ensure_mixing_distribution()
            self._mixing_distribution.boom()
            self._boom_model = boom.FiniteMixtureModel(
                boom_component_list.values,
                self._mixing_distribution.boom()
            )

            sampler = boom.FiniteMixturePosteriorSampler(self._boom_model)
            self._boom_model.set_method(sampler)
            self._assign_data_to_boom_model(
                self._boom_model,
                self._mixture_components[0])

        return self._boom_model

    def plot_loglike(self, burn=0, fig=None, ax=None, style="ts", **kwargs):
        style = unique_match(style, ["ts", "histogram", "density"])

        fig, ax = R.ensure_ax(fig, ax)
        if burn < 0:
            burn = 0

        niter = self._log_likelihood_draws.shape[0]

        iteration = range(burn, niter)

        if style == "ts":
            ax.plot(iteration, self._log_likelihood_draws[burn:])
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Log likelihood")

        elif style == "histogram":
            R.hist(self._log_likelihood_draws[burn:], ax=ax)
            ax.set_xlabel("Log likelihood")

        elif style == "density":
            den = R.Density(self._log_likelihood_draws[burn:])
            den.plot(ax=ax)
            ax.set_xlabel("Log likelihood")
            ax.set_ylabel("density")

        return fig, ax

    def plot_components(self, burn=0, style="ts", fig=None, ax=None, **kwargs):
        fig, ax = self._mixture_components[0].plot_components(
            self._mixture_components,
            burn=burn,
            style=style,
            fig=fig,
            ax=ax,
            **kwargs)

        return fig, ax

    def plot_mixing_weights(self, burn=0, style="boxplot", fig=None, ax=None, **kwargs):
        fig, ax = R.ensure_ax(fig, ax)
        style = R.unique_match(style, ["boxplot", "ts", "density"])
        if burn < 0:
            burn = 0
        probs = self._mixing_distribution._prob_draws[burn:, :]
        niter = self._mixing_distribution._prob_draws.shape[0]
        S = probs.shape[1]

        if style == "ts":
            iteration = range(burn, niter)
            for s in range(S):
                ax.plot(iteration, probs[:, s], label=s)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Mixing Weights")
            ax.legend()

        elif style == "boxplot":
            ax.boxplot(probs)
            ax.set_xlabel("Mixture Component")
            ax.set_ylabel("Mixing Weights")

        elif style == "density":
            for s in range(S):
                den = R.Density(probs[:, s])
                den.plot(ax=ax, label=s)
            ax.set_xlabel("Mixing Weights")
            ax.set_ylabel("density")
            ax.legend()

        return fig, ax


    def _assign_data_to_boom_model(self, boom_model, mixture_component):
        data_builder = mixture_component.create_boom_data_builder(self._data)
        boom_data = data_builder.build_boom_data(self._data)
        for data_point in boom_data:
            boom_model.add_data(data_point)

    def _ensure_mixing_distribution(self):
        if self._mixing_distribution is None:
            dim = self.num_components
            uniform_probs = np.ones(dim) / dim
            self._mixing_distribution = R.MultinomialModel(uniform_probs)

            self._mixing_distribution.set_prior(
                R.DirichletPrior(uniform_probs * 1.0))



    def _allocate_space(self, niter):
        for component in self._mixture_components:
            component.allocate_space(niter)
        self._ensure_mixing_distribution()
        self._mixing_distribution.allocate_space(niter)

        self._log_likelihood_draws = np.empty(niter)
        self._class_probs = np.empty(
            (niter, self.sample_size, self.num_components)
        )

    def _record_draw(self, iteration):
        for component in self._mixture_components:
            component.record_draw(iteration)
        self._mixing_distribution.record_draw(iteration)

        self._log_likelihood_draws[iteration] = self._boom_model.last_loglike
        self._class_probs[iteration, :, :] = (
            self._boom_model.class_membership_probability.to_numpy()
        )
