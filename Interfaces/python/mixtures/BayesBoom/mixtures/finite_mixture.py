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

    def _assign_data_to_boom_model(self, boom_model, mixture_component):
        data_builder = mixture_component.create_boom_data_builder()
        boom_data = data_builder.build_boom_data(self._data)
        for data_point in boom_data:
            boom_model.add_data(data_point)
    
    def _ensure_mixing_distribution(self):
        if self._mixing_distribution is None:
            dim = self.num_components
            uniform_probs = np.ones(dim) / dim
            self._mixing_distribution = R.MultinomialModel(uniform_probs)
    
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
