import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import BayesBoom.boom as boom
import BayesBoom.spikeslab as spikeslab
import BayesBoom.R as R

from .state_models import StateModel

import matplotlib.pyplot as plt
import copy
import patsy


class Bsts:
    """
    Bayesian structural time series models.  Any other "BS" is the fault of the
    analyst!!

    Bsts supports models for scalar time series, where the outcome variable is
    either conditionally Gaussian, student T, binomial (logit linke), or
    Poisson (log link).  A key feature of bsts models is optional support for
    contemporaneous covariates through a Bayesian spike-and-slab prior that
    handles model selection, averaging, and uncertainty.

    This class is a port of the bsts R package.  The underlying C++ code for
    the two packages is the same, but the interfaces are slightly different to
    reflect seemingly "natural" approaches in the two languages.

    Expected usage:
    data = get_time_series_from_somewhere()  # A pd.Series
    model = Bsts()
    model.add_state(LocalLinearTrend(data))
    model.add_state(SeasonalStateModel(data))
    model.train(niter=1000)

    model.plot()
    model.plot("coef")
    model.plot("coefficients")
    model.plot("comp")

    pred = model.predict(12)
    pred.plot()
    pred.posterior_mean()
    """

    def __init__(self, family: str = "gaussian", seed: int = None):
        """
        Create an "empty" bsts model.

        Args:

          family: The type of error distribution for the outcome variable.
            This will eventually support gaussian, poisson, student, and
            binomial errors.  For now only "gaussian" is supported, but the
            others are on the roadmap.

          seed:
            An integer (or None) containingt the random seed for the C++ random
            number generator.

        """
        # The model family.
        supported_families = ["gaussian", "student", "binomial", "poisson"]
        self._family = R.unique_match(family.lower(), supported_families)

        # self._model is the BayesBoom handle to the BOOM C++ state space model
        # object.
        self._model = None

        # The 'observation model' lives inside self._model.  The job of the
        # observation model manager is to handle specific cases of observation
        # models in terms of housekeeping like recording MCMC draws.
        self._observation_model_manager = None

        # self._state_models is a list of StateModel objects that reflect the
        # state models in self._model.  Their job is similar to that of the
        # observation_model_manager.
        self._state_models = []

        # The dimension of the latent state.
        self._state_dimension = 0

        # For the random number generator to work correctly, a python object
        # must hold a copy of the BOOM global random number generator.
        # Otherwise it goes out of scope and gets freshly re-initialized each
        # time.
        self._rng = boom.GlobalRng.rng
        if seed is not None:
            self._rng.seed(seed)

    def add_state(self, state_model: StateModel):
        """
        Add a component of state to the model.  This is typically done before
        training.

        Args:
          state_model: A StateModel object describing the state to be added.
            Examples of class StateModel include LocalLevel and
            LocalLinearTrend models for trend, Seasonal or Trig state models
            for modeling seaonality, and various holiday state models for
            describing holiday effects.

        Effects:
          The state model is appended to self._state_models, and the state
          model being added is is informed about the position of the state it
          manages in the global state vector.
        """
        self._state_models.append(state_model)
        state_model.set_state_index(self._state_dimension)
        self._state_dimension += state_model.state_dimension

    def train(self, data, niter: int, formula=None, prior=None,
              ping: int = None, **kwargs):
        """
        Train a bsts model by running a specified number of MCMC iterations.

        Args:
          formula: Either numeric time series (array-like), or a string giving
            a formula that can be interpreted by the 'patsy' package (python's
            version of R's model syntax).
          data: If a formula is given, data is a DataFrame containing the
            variables from the formula.  If 'formula' is a numeric then 'data'
            need not be specified.
          prior: The prior distribution for the observation model.  If a
            regression component is included then this is a
            spikeslab.RegressionSpikeSlabPrior describing the regression
            coefficients and the residual standard deviation.  Otherwise it is
            a boom.SdPrior on the residual standard deviation.  If None then a
            default prior will be chosen.
          niter:  The desired number of MCMC iterations.
          ping: The frequency with which to print status updates in the MCMC
            algorithm.  The default is niter/10.  If ping <= 0 then no status
            updates are printed.
          **kwargs: Extra arguments are passed to self._create_model, and used
            to create the prior distribution in the event that the 'prior' arg
            is None.  Extra arguments must be recognizable to the code that
            creates the priors, or an exception is raised.

        Effects:
          self._model: created, and populated with MCMC draws.
          self._state_models: MCMC draws for state model parameters are stored
            in the state model objects held by self._state_models.  The
            contribution of each state_model to the overall state are likewise
            stored in the state_model objects.
          self._observation_model_manager: Model parameters and summaries
            relevant to the observation model are stored in

        The following objects are recorded in self.
          self._original_series: the time series used in the training run.
          self._prior: the prior used in the training run.
          self._formula: the fomula used to specify the regression component.
        """
        self._create_model(self._family, formula, data, prior, **kwargs)
        # One call to sample_posterior is needed prior to allocating space, to
        # set up dynamic memory allocations on the C++ side.
        self._model.sample_posterior()
        self._allocate_space(niter)

        for i in range(niter):
            self._model.sample_posterior()
            self._record_draws(i)

        return None

    @property
    def state_dimension(self):
        """
        The dimension of the state vector.
        """
        return self._state_dimension

    @property
    def niter(self):
        """
        The number of MCMC iterations that have been run.
        """
        if self._observation_model_manager is None:
            return 0
        return self._observation_model_manager.niter

    @property
    def time_dimension(self):
        """
        The number of time points that have been observed, including any time
        points where the time series variable was missing.
        """
        return 0 if self._model is None else self._model.time_dimension

    @property
    def number_of_state_models(self):
        return len(self._state_models)

    @property
    def original_series(self):
        """
        The target series in the model training step.
        """
        if hasattr(self, "_original_series"):
            return self._original_series
        else:
            return None

    @property
    def predictors(self):
        """
        The predictors used in the training data.
        """
        if (
                self._formula is None
                or not hasattr(self, "_data")
                or self._data is None
        ):
            return None
        if "~" in self._formula:
            predictor_formula = self._formula.split("~")[1]
        else:
            predictor_formula = self._formula
        return patsy.dmatrix(predictor_formula, self._data)

    def coefficients(self, burn=None):
        """
        The posterior distribution of the model coefficients.  If the model
        contains no regression component, return None.

        Args:
          burn: The number of MCMC iterations to be discarded as burn-in.  If
            None then a default value is chosen.  Set burn=0 if no burn-in is
            desired.

        Returns:
          A scipy.sparse.lil_matrix containing the draws of the model
          coefficients.  Rows are MCMC draws.  Columns correspond to predictor
          variables.
        """
        if burn is None:
            burn = self.suggest_burn()
        if self._observation_model_manager.has_regression:
            return self._observation_model_manager._coefficients[burn:, :]
        else:
            return None

    def inclusion_probabilities(self, burn=None):
        """
        Args:
          burn: The number of MCMC iterations to be discarded as burn-in.  If
            None then a default value is chosen.  Set burn=0 if no burn-in is
            desired.

        Returns:
          A pd.Series indexed by predictor names, containing the marginal
          inclusion probabilities of each coefficient.
        """
        coef = self.coefficients(burn=burn)
        if coef is None:
            return None
        inc = spikeslab.compute_inclusion_probabilities(coef)
        return pd.Series(inc, index=self._predictor_names)

    def residuals(self, burn=None):
        """
        The residuals are the smoothing residuals from the model.  I.e. the
        residuals when you get to look both forward and backward in time.
        """
        if burn is None:
            burn = self.suggest_burn()
        y = R.to_numpy(self.original_series).reshape((1, -1))
        yhat = self.state_contribution[burn:, :]
        if self._observation_model_manager.has_regression:
            contrib = self._observation_model_manager.regression_contribution
            yhat += contrib[burn:, :]
        return y - yhat

    @property
    def state_contribution(self):
        """
        The posterior distribution of the mean at each time point, as determined
        by the simulated state, without observation noise.
        """
        contribution = np.zeros((self.niter, self.time_dimension))
        for model in self._state_models:
            contribution += model._state_contribution
        return contribution

    def one_step_prediction_errors(self,
                                   cutpoints=None,
                                   burn=None,
                                   standardize=False,
                                   simplify=True):
        """
        The posterior distribution of the one-step-ahead prediction errors from
        the model training.  The errors are computing using the Kalman filter,
        and are of two types.

        Purely in-sample errors are computed as a by-product of the Kalman
        filter as a result of fitting the model.  These are stored in the
        bsts.object assuming the save.prediction.errors argument is TRUE, which
        is the default.  The in-sample errors are 'in-sample' in the sense that
        the parameter values used to run the Kalman filter are drawn from their
        posterior distribution given complete data.  Conditional on the
        parameters in that MCMC iteration, each 'error' is the difference
        between the observed y[t] and its expectation given data to t-1.

        Purely out-of-sample errors can be computed by specifying the
        'cutpoints' argument.  If cutpoints are supplied then a separate MCMC
        is run using just data up to the cutpoint.  The Kalman filter is then
        run on the remaining data, again finding the difference between y[t]
        and its expectation given data to t-1, but conditional on parameters
        estimated using data up to the cutpoint.

        Args:
          cutpoints: An increasing sequence of integers between 1 and the
            number of time points in the training data, or None.  If None then
            the in-sample one-step prediction errors will be extracted and
            returned.  Otherwise the model will be re-fit with a separate MCMC
            run for each entry in 'cutpoints'.  Data up to each cutpoint will
            be included in the fit, and one-step prediction errors for data
            after the cutpoint will be computed.
          burn:  The number of MCMC iterations to discard as burn-in.
          standardize: (bool)  If True then the prediction errors are divided
            by the square root of the one-step-ahead forecast variance.  If
            False the raw errors are returned.
          simplify: If True, and only one set of prediction errors was
            requested, then return just that set as a numpy array.

        Returns:
          A dict, keyed by cutpoint values, with entries giving the
          distribution of one-step prediction errors corresponding to
          individual cutpoints.  Each list entry is a matrix, with rows
          corresponding to MCMC draws, and columns corresponding to time points
          in the data for bsts.object.  If the in-sample prediction errors were
          stored in the original model fit, they will be present in the output.
        """
        has_prediction_errors = hasattr(self, "_one_step_prediction_errors")
        if burn is None:
            burn = self.suggest_burn()
        if cutpoints is None and has_prediction_errors:
            if simplify:
                return self._one_step_prediction_errors[
                    self.time_dimension][burn:, :]
            else:
                return {
                    self.time_dimension:
                    self._one_step_prediction_errors[
                        self.time_dimension][burn:, :]
                }

        elif cutpoints is not None and has_prediction_errors:
            required_cutpoints = [
                x for x in cutpoints
                if x not in self._one_step_prediction_errors.keys()
            ]

            if required_cutpoints:
                extra_prediction_errors = (
                    self._model.compute_prediction_errors(
                        self.niter,
                        required_cutpoints,
                        standardize)
                )
                for i, cutpoint in enumerate(required_cutpoints):
                    errors = extra_prediction_errors[i].to_numpy()
                    self._one_step_prediction_errors[cutpoint] = errors

            ans = {
                cut: self._one_step_prediction_errors[
                    cut][burn:, :] for cut in cutpoints
            }
            return copy.deepcopy(ans)

        else:
            raise Exception(
                "Cannot find one_step_prediction_errors.  The errors are not "
                "available for logit and Poisson models, and are only "
                "available for ther models after model training.")

    @property
    def log_likelihood(self):
        """
        The vector of log liklelihood values associated with the MCMC run.  This
        only exists if the model is Gaussian.  Otherwise None is returned.
        """
        if (
                hasattr(self, "_observation_model_manager")
                and hasattr(self._observation_model_manager,
                            "_log_likelihood")
        ):
            return self._observation_model_manager._log_likelihood
        else:
            return None

    def suggest_burn(self):
        """
        Suggest a number of burn-in iterations.  For Gaussian models this will
        be based on the simulated values of log-likelihood.  For other models
        it will be a fixed percentage of the draws.
        """
        loglike = self.log_likelihood
        if loglike is None:
            burn = self.niter / 10
        else:
            burn = R.suggest_burn(loglike)
        return int(burn)

    def predict(self, newdata, burn: int = None, seed: int = None,
                separate_components=False, **kwargs):
        """
        Args:
          newdata: For regression models 'newdata' is a DataFrame containing
            the predictor variables to use in the regression.  The forecast
            horizon (the number of periods to predict) is the number of rows in
            'newdata'.  For pure time series models, 'newdata' is the
            integer-valued forecast horizon.
          burn: The number of MCMC interations to discard as burnin.  If None
            then an attempt will be made to select a reasonable value.
          seed: An integer used to seed the C++ random number generator driving
            the simulations from the posterior predictive distribution.  If
            None then the current state of the RNG will be used.  The 'seed'
            argument is mainly useful for reproducibility when testing.
          separate_components: If True then an extra dimension is added to the
            output, and the contributions of each state component are kept
            apart.
          **kwargs: Additional named arguments are passed to the 'predict'
            method of the ObservationModelManager specific to the observation
            model.  The main use is to supply 'trials' for logit models, and
            'exposure' for Poisson models.

        Returns:
            A BstsPrediction object containing the predictions.
        """
        if burn is None:
            burn = self.suggest_burn()
        burn = int(burn)

        if seed is not None:
            self._rng.seed(int(seed))

        ndraws = self.niter - burn
        manager = self._observation_model_manager
        formatted_prediction_data = manager.format_prediction_data(
            newdata, **kwargs)
        self.predictor_names = formatted_prediction_data.get("xnames", None)
        horizon = formatted_prediction_data["forecast_horizon"]
        if separate_components:
            nstate = self._model.number_of_state_models
            pred = np.empty((ndraws, nstate + 2, horizon))
        else:
            pred = np.empty((ndraws, horizon))

        total_time_points = len(self.original_series) + horizon
        for state_model in self._state_models:
            state_model.observe_time_dimension(total_time_points)

        for i in range(burn, self.niter):
            self._restore_draw(i)
            if separate_components:
                pred[i - burn, :, :] = self._observation_model_manager.predict(
                    self._model,
                    formatted_prediction_data,
                    boom.Vector(self._final_state[i, :]),
                    rng=self._rng,
                    separate_components=True,
                )
            else:
                pred[i - burn, :] = self._observation_model_manager.predict(
                    self._model,
                    formatted_prediction_data,
                    boom.Vector(self._final_state[i, :]),
                    rng=self._rng,
                    separate_components=False
                )

        return BstsPrediction(pred, self.original_series)

    def help(self):
        # import webbrowser as web.
        # web.open("path_to_docs")
        print("coming soon...")

    # --------------------------------------------------------------------------
    # The remaining public functions all make plots.
    # --------------------------------------------------------------------------
    def plot(self, what: str = None, **kwargs):
        """
        Dispatch a plot request to a specific plotting function.

        Args:
          what: The type of plot desired.  This can be any initial string that
            unambiguously matches the set of plot types listed below.
          **kwargs: Additional named argument expected by the specific plotting
            functions.

        Returns:
          Varies by function.  Usually the figure or axes object (or both) used
          to draw the plot.
        """
        plot_types = ["state", "components", "coefficients", "inclusion",
                      "residuals", "prediction_errors",
                      "forecast_distribution", "predictors",
                      "size", "dynamic", "seasonal", "monthly",
                      "help"]
        if what is None:
            what = "state"
        what = R.unique_match(what.lower(), plot_types)
        if what == "state":
            return self.plot_state(**kwargs)
        elif what == "components":
            return self.plot_state_components(**kwargs)
        elif what == "coefficients":
            return self.plot_coefficients(**kwargs)
        elif what == "inclusion":
            return self.plot_inclusion_probs(**kwargs)
        elif what == "residuals":
            return self.plot_residuals(**kwargs)
        elif what == "prediction_errors":
            return self.plot_prediction_errors(**kwargs)
        elif what == "forecast_distribution":
            return self.plot_forecast_distribution(**kwargs)
        elif what == "predictors":
            return self.plot_predictors(**kwargs)
        elif what == "size":
            return self.plot_model_size(**kwargs)
        elif what == "dynamic":
            return self.plot_dynamic_regression(**kwargs)
        elif what == "seasonal":
            return self.plot_seasonal(**kwargs)
        elif what == "monthly":
            return self.plot_monthly(**kwargs)
        elif what == "help":
            return self.plotting_help()
        else:
            raise Exception(f"Don't know how to plot {what}.")

    def plot_state(self,
                   burn: int = None,
                   time=None,
                   show_actuals: bool = True,
                   style: str = None,
                   scale: str = None,
                   ylim: tuple = None,
                   ax: plt.Axes = None,
                   **kwargs):
        """
        The evolving filtered and smoothed distribution of the posterior mean of
        the outcome variable over time, along with the observed time points.

        Args:
          burn: The number of MCMC iterations to discard as burn-in.  If None
            then an attempt will be made to use a reasonable default.
          time: A set of timestamps, which must be the same length as
            self.original_series.
          show_actuals: Should the original series be plotted (as points) on
            top of the posterior mean?
          style: "dynamic" means to plot the distribution using a dynamic
            distribution plot.  "boxplot" means to use a time series boxplot.
          scale: "linear" or "mean".  For Poisson or logit models involving a
            link function the "linear" scale is the scale of the linear
            predictor in a generalized linear model (e.g. the log scale for
            Poisson, or logit scale for logit models).  The "mean" scale plots
            the curve on the scale of the observed data (the mean of the
            Poisson, or the probability scale for logit models).
          ylim: (lower, upper) limits to use for the vertical axis.
          ax: The plt.Axes object on which to draw the plot.  If None then a
            new figure and axes will be created.
          **kwargs: Extra arguments passed to plot_dynamic_distribution or
            time_series_boxplot.
        """
        if style is None:
            style = "dynamic"
        style = R.unique_match(style, ["dynamic", "boxplot"])

        if scale is None:
            scale = "linear"
        scale = R.unique_match(scale, ["linear", "mean"])

        niter = self.niter
        if burn is None:
            burn = self.suggest_burn()

        if time is None:
            time = self.original_series.index

        state_contribution = np.zeros((niter, len(time)))
        for model in self._state_models:
            state_contribution += model._state_contribution

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = None

        if style == "dynamic":
            R.plot_dynamic_distribution(
                curves=state_contribution,
                timestamps=time,
                ax=ax,
                ylim=ylim,
                **kwargs)

        elif style == "boxplot":
            R.time_series_boxplot(
                curves=state_contribution,
                time=time,
                ax=ax,
                ylim=ylim,
                **kwargs)

        if show_actuals:
            ax.scatter(time,
                       self.original_series,
                       s=10 / np.sqrt(len(self.original_series)))
        return fig, ax

    def plot_state_components(self,
                              burn: int = None,
                              time: np.array = None,
                              same_scale: bool = True,
                              ylim: tuple = None,
                              fig=None,
                              components=None,
                              **kwargs):
        """
        Plot the contribution of each state model.

        Args:

          burn: The number of MCMC iterations to discard as burn-in.  If None
            then an attempt will be made to use a reasonable default.
          time: A set of timestamps, which must be the same length as
            self.original_series.
          same_scale: Should all the plots share the same scale.  If False then
            each plot has a different scale centered around its range of
            variation.  Ignored if 'ylim' is provided.
          ylim: (lower, upper) limits to use for the vertical axis.  This
            applies to all axes in the plot.
          fig: A plt.Figure object on which to draw the subplots.  If None then
            one will be generated.
          components: An iterable that can be used to index the state models.
          **kwargs: Extra args passed to the 'plot_state_contribution' method
            of each state model.

        Returns:
          The plt.Figure object on which the plots are drawn.

        """
        if fig is None:
            fig = plt.figure(constrained_layout=True)
        if components is None:
            state_models = self._state_models
        else:
            state_models = self._state_models[components]

        if burn is None:
            burn = self.suggest_burn()
        elif burn < 0:
            burn = 0

        if time is None:
            time = self.original_series.index

        if same_scale is True and ylim is None:
            ylim = _find_state_contribution_ylim(state_models, burn)

        # for documentation about how gridspec works see
        # https://matplotlib.org/3.1.1/tutorials/intermediate/gridspec.html
        has_regression = self._observation_model_manager.has_regression
        number_of_contributions = len(state_models) + has_regression
        nr, nc = R.plot_grid_shape(number_of_contributions)
        outer_grid = fig.add_gridspec(nr, nc)
        plot_index = 0
        need_regression = has_regression
        for i in range(nr):
            for j in range(nc):
                if plot_index < len(state_models):
                    state_model = state_models[plot_index]
                    state_model.plot_state_contribution(
                        fig=fig,
                        gridspec=outer_grid[i, j],
                        time=time,
                        burn=burn,
                        ylim=ylim,
                        **kwargs)
                elif need_regression:
                    self._observation_model_manager.plot_regression_contribution( # noqa
                        fig=fig,
                        gridspec=outer_grid[i, j],
                        time=time,
                        burn=burn,
                        ylim=ylim,
                        **kwargs)
                plot_index += 1
        return fig

    def plot_coefficients(self, burn=None, inclusion_threshold=0,
                          unit_scale=True,
                          number_of_variables=None, ax=None, **kwargs):
        coef = getattr(self._observation_model_manager,
                       "_coefficients", None)
        if coef is None:
            raise Exception("Model has no coefficients.")

        if burn is None:
            burn = self.suggest_burn()

        return spikeslab.plot_inclusion_probs(
            coef, burn=burn, xnames=self._predictor_names,
            inclusion_threshold=inclusion_threshold,
            unit_scale=unit_scale, number_of_variables=number_of_variables,
            ax=ax, **kwargs)

    def plot_inclusion_probs(self, **kwargs):
        return self.plot_coefficients(**kwargs)

    def plot_residuals(self, burn=None, time=None, style="dynamic", means=True,
                       ax=None, **kwargs):
        """
        Plots the posterior distribution of the residuals from the bsts
        model, after subtracting off the state effects (including
        regression effects).

        Args:
          burn: The number of MCMC iterations to be discarded as burn-in.  If
            None then a default value will be used.
          time: An optional vector of values to plot on the time axis.
          means: If True then the posterior mean of each residual is plotted as
            a dot on top of the boxplot or the dynamic distribution plot.
          style: Either "dynamic", for dynamic distribution plots, or
            "boxplot", for box plots.  Partial matching is allowed, so
            "dyn" or "box" would work, for example.
          ax: The plt.Axes object on which to draw the plot.  If None then a
            new figure and axes will be created.
          kwargs:  Extra arguments passed to plot_dynamic_distribution.

        Returns:
          This function is called for its side effect, which is to
          produce a plot on the current graphics device.
        """
        style = R.unique_match(style.lower(), ["boxplot", "dynamic"])
        if time is None:
            time = self.original_series.index

        residuals = self.residuals(burn)

        if style == "dynamic":
            R.plot_dynamic_distribution(
                curves=residuals,
                timestamps=time,
                ax=ax,
                **kwargs)

        elif style == "boxplot":
            R.time_series_boxplot(
                curves=residuals,
                time=time,
                ax=ax,
                **kwargs)

        return ax

    def plot_forecast_distribution(
            self,
            cutpoints=None,
            burn=None,
            style="dynamic",
            xlab="Time",
            ylab="",
            main="",
            show_actuals=True,
            cex_actuals=1,
            col_actuals="blue", fig=None, **kwargs):
        """
        Plots the posterior distribution of the one-step-ahead forecasts
        for a bsts model.

        Args:
          cutpoints: A list of integers, or None.
          burn:  The number of MCMC iterations to be discarded as burn-in.
          style: "dynamic" or "boxplot" indicating whether dynamic distribution
            plots or boxplots should be used to draw the evolving distributions.
          xlab:  Label for the time axis.
          ylab:  Label for the vertical axis.
          main:  Main plot title.
          show_actuals: If True, then the actual response values will be
            plotted on top of the forecast distributions.
          col_actuals:  The color to use for the actual values, if shown.
          fig:  The plt.Figure on which to draw the plot.
          **kwargs: Extra args passed to either plot_dynamic_distribution, or
            time_series_boxplot.

        Returns:
          The plt.Figure on which the plot is drawn.
        """
        errors = self.one_step_prediction_errors(
            cutpoints=cutpoints, burn=burn, simplify=False)
        if len(errors) == 0:
            raise Exception("No forecast errors available.")
        forecast = errors
        original = self.original_series.values.reshape(1, -1)
        for key in forecast.keys():
            forecast[key] = original - errors[key]
        timestamps = self.original_series.index
        actuals = self.original_series if show_actuals is True else None

        implied_cutpoints = list(errors.keys())
        vertical_cuts = [np.NaN] * len(implied_cutpoints)
        for i in range(len(vertical_cuts)):
            if implied_cutpoints[i] < len(timestamps):
                vertical_cuts[i] = timestamps[implied_cutpoints[i]]

        fig = R.compare_dynamic_distributions(
            [forecast[key] for key in implied_cutpoints],
            timestamps=timestamps,
            style=style,
            xlab=xlab,
            ylab=ylab,
            main=main,
            fig=fig,
            actuals=actuals,
            cex_actuals=cex_actuals,
            col_actuals=col_actuals,
            vertical_cuts=vertical_cuts)

        return fig

    def plot_prediction_errors(self, cutpoints=None, burn=None, xlab="Time",
                               ylab="", main="", fig=None, **kwargs):
        prediction_errors = self.one_step_prediction_errors(
            cutpoints=cutpoints, burn=burn)
        timestamps = self.original_series.index
        R.compare_dynamic_distributions(
            prediction_errors,
            timestamps=timestamps,
            xlab=xlab,
            ylab=ylab,
            main=main,
            fig=fig,
            actuals=None,
            vertical_cuts=list(cutpoints) + [self.time_dimension],
            **kwargs)
        return prediction_errors

    def plot_predictors(self,
                        burn=None,
                        inclusion_threshold=.1,
                        ylim=None,
                        flip_signs=True,
                        show_legend=True,
                        grayscale=True,
                        short_names=False,
                        ax=None,
                        **kwargs):
        """
        Plot the predictors with sufficiently high inclusion probability vs
        time, along with the outcome.

        Args:
          burn:  The number of MCMC iterations to discard as burn-in.
          inclusion_threshold: An inclusion probability that coefficients
            must exceed in order to be displayed.
          ylim:  Limits on the vertical axis.
          flip_signs: If True then a predictor with a negative sign will
            be flipped before being plotted, to better align visually
            with the target series.
          show_legend:  Show a legend for the plot.
          grayscale: If True then the plotted lines are shown with "blackness"
            corresponding to their inclusion probability.  Certain predictors
            appear as black lines.  Improbable ones are fainter.
          short_name: If True then common prefixes and suffixes are removed
            from variable names in the legend.
          ax: a plt.Axes object on which to draw the plot.
          **kwargs:  Extra arguments passed to 'R.plot_ts'.

        Returns:
          The 'ax' on which the plot was drawn.
        """
        coef = getattr(self._observation_model_manager, "_coefficients", None)
        if coef is None:
            return ax

        if burn is None:
            burn = self.suggest_burn()
        if burn > 0:
            coef = coef[burn:, :]

        if ax is None:
            _, ax = plt.subplots(1, 1)

        inc = spikeslab.compute_inclusion_probabilities(coef)

        keep = inc >= inclusion_threshold
        if not np.any(keep):
            ax.set_title("No predictors above inclusion threshold.")
            return ax

        inc = inc[keep]

        predictors = self.predictors[:, keep]
        pos = spikeslab.coefficient_positive_probability(predictors)
        order = np.argsort(inc)[::-1]
        predictors = predictors[:, order]
        inc = inc[order]
        pos = pos[order]

        for i in range(predictors.shape[1]):
            sd = np.nanstd(predictors[:, i], ddof=1)
            mean = np.nanmean(predictors[:, i])
            predictors[:, i] = (predictors[:, i] - mean) / sd

        if grayscale:
            line_colors = (1 - inc).astype("str")
        else:
            line_colors = ["black"] * len(inc)

        number_of_predictors = predictors.shape[1]
        times = self.original_series.index
        predictor_names = np.array(self._predictor_names)[keep][order]
        if short_names:
            predictor_names = R.remove_common_prefix(predictor_names)
            predictor_names = R.remove_common_suffix(predictor_names)

        scaled_original = self.original_series - np.nanmean(
            self._original_series)
        scaled_original /= np.nanstd(scaled_original, ddof=1)
        original_name = self._formula.split("~")[0]
        ax.plot(times, scaled_original, label=original_name)

        for i in range(number_of_predictors):
            ax.plot(times,
                    predictors[:, i],
                    label=predictor_names[i],
                    linestyle=R.lty(i),
                    color=line_colors[i],
                    lw=.5,
                    **kwargs)

        if show_legend:
            ax.legend()

        return ax

    def plot_size(self, burn=None, ax=None, **kwargs):
        coef = getattr(self._observation_model_manager,
                       "_coefficients", None)
        if coef is None:
            raise Exception("Model has no coefficients.")
        if burn is None:
            burn = self.suggest_burn()
        if ax is None:
            _, ax = plt.subplots(1, 1)
        return spikeslab.plot_model_size(
            coef, burn=burn, ax=ax, **kwargs)

    def plot_dynamic_regression(self, **kwargs):
        pass

    def plot_seasonal(self, nseasons=None, season_duration=None,
                      same_scale=True, ylim=None, get_season_name=None,
                      burn=None, fig=None, **kwargs):
        """
        Plot one or more seasonal state components as "monthplots"

        Args:
          nseasons: Plot the seasonal state component with this many seasons.
            If no such model exists then raise an error.  If None, then all
            seasonal models are plotted.
          season_duration: If there are multiple seasonal models with the same
            'nseasons', plot the seasonal model with this season_duration.
          same_scale: If True then the seasonal effects are plotted with a
            common Y axis scale.  If False then each plot determines its own Y
            axis scale.
          ylim: A pair of values giving the lower and upper limits on the Y
            axis.  If supplied, this forces same_scale to True.
          get_season_name: A function or callable object that takes a timestamp
            and returns a string that can be used as a plot label.  If None and
            nseasons is one of the special values listed below, then the
            associated function will be used.
            - 4  R.quarters
            - 7  R.weekdays
            - 12 R.months
          burn: The nubmer of MCMC iterations to be discarded as burn-in.  If
            None then a value will be suggested using suggest_burn.
          fig: The plt.Figure object on which to draw the plot.  If None then
            an object will be created and returned.
          **kwargs: Extra arguments will be passed to
            R.plot_dynamic_distribution.

        Returns:
          The plt.Figure object on which the plot is drawn.
        """
        from .seasonal import SeasonalStateModel

        if fig is None:
            fig = plt.figure()
        state_models = [x for x in self._state_models
                        if isinstance(x, SeasonalStateModel)]
        if nseasons is not None:
            state_models = [x for x in state_models if x.nseasons == nseasons]

        if season_duration is not None and len(state_models) > 1:
            state_models = [x for x in state_models
                            if x.season_duration == season_duration]

        if len(state_models) == 0:
            raise Exception("No suitable SeasonalStateModel objects found.")

        num_plots = len(state_models)
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        return fig

    def plot_monthly(self, **kwargs):
        pass

    def plotting_help(self, **kwargs):
        # TODO, once we get documentation up on "readthedocs", uncomment
        # import webbrowser as web
        # web.open("path/to/plot/function")
        #
        print("coming soon.")

    def _allocate_space(self, niter: int):
        """
        Allocate space in the model for 'niter' MCMC draws.
        """
        self._observation_model_manager.allocate_space(
            niter, self.time_dimension)
        for state_model in self._state_models:
            state_model.allocate_space(niter, self.time_dimension)
        self._final_state = np.empty((niter, self.state_dimension))

        # self._one_step_prediction_errors is a dict keyed by the index of the
        # last time point in the in sample trainind data.  During initial model
        # training errors are computed only the full training data.  If
        # desired, errors can be computed for other cutpoints later, by calling
        # self.one_step_prediction_errors().
        self._one_step_prediction_errors = {
            self.time_dimension: np.empty((niter, self.time_dimension))
        }

    def _record_draws(self, iteration: int):
        """
        Record the parameters and state from each state model.

        Args:
          iteration: The iteration (MCMC draw) number being recorded.
        """
        state_matrix = self._model.state.to_numpy()
        for m in self._state_models:
            m.record_state(iteration, state_matrix)
        self._observation_model_manager.record_draw(
            iteration, self._model)
        self._final_state[iteration, :] = state_matrix[:, -1]
        self._one_step_prediction_errors[self.time_dimension][iteration, :] = (
            self._model.one_step_prediction_errors(False).to_numpy()
        )

    def _restore_draw(self, iteration: int):
        self._observation_model_manager.restore_draw(
            iteration, self._model)
        for state_model in self._state_models:
            state_model.restore_state(iteration)

    def _create_model(self, family, formula, data, prior):
        """Create the boom model object.

        Args:
          formula: Either numeric time series (array-like), or a string giving
            a formula that can be interpreted by the 'patsy' package (python's
            version of R's model syntax).
          data: If a formula is given, data is a DataFrame containing the
            variables from the formula.  If 'formula' is a numeric then 'data'
            need not be specified.
          prior: The prior distribution for the observation model.  If a
            regression component is included then this is a
            spikeslab.RegressionSpikeSlabPrior describing the regression
            coefficients and the residual standard deviation.  Otherwise it is
            a boom.SdPrior on the residual standard deviation.  If None then a
            default prior will be chosen.

        Effects:
          self._model is created, populated with data and assigned a posterior
            sampler.
          self._original_series is populated with the time series being modeled.
          The data argument is stored in self._data.

        Return:
          self._model
        """
        factory = StateSpaceModelFactory.create(family, formula)
        self._formula = formula
        self._model = factory.create_model(prior, data, self._rng)
        self._prior = factory._prior
        self._predictor_names = factory.predictor_names
        self._data = data
        if hasattr(factory, "_original_series"):
            self._original_series = factory._original_series
            if isinstance(self._original_series, np.ndarray):
                self._original_series = pd.Series(self._original_series.ravel())

        self._observation_model_manager = (
            factory.create_observation_model_manager()
        )
        for state_model in self._state_models:
            self._model.add_state(state_model._state_model)
        return self._model

    def __setstate__(self, payload):
        """
        How to un-pickle the model
        """
        self.__init__(family=payload["family"])

        state_models = payload["state_models"]
        for state in state_models:
            self.add_state(state)

        niter = payload.get("niter", None)
        data = payload.get("data", None)
        if isinstance(data, int):
            data = payload.get("original_series", None)

        if niter and niter > 0:
            self._formula = payload.get("formula", None)
            self._prior = payload.get("prior", None)
            self._model = self._create_model(
                self._family,
                self._formula,
                data=data,
                prior=payload.get("prior", None))
            self._observation_model_manager = payload[
                "observation_model_manager"]
            self._original_series = payload["original_series"]
            self._final_state = payload["final_state"]

    def __getstate__(self):
        """
        Return a dict full of picklable entries fully describing the object
        state.
        """
        payload = {
            "family": self._family,
            "observation_model_manager": self._observation_model_manager,
            "state_models": self._state_models,
        }

        if hasattr(self, "_formula"):
            payload["formula"] = self._formula
        if hasattr(self, "_prior"):
            payload["prior"] = self._prior
        payload["niter"] = self.niter
        if hasattr(self, "_original_series"):
            payload["original_series"] = self._original_series
        if hasattr(self, "_data"):
            payload["data"] = self._data
        if hasattr(self, "_final_state"):
            payload["final_state"] = self._final_state

        return payload


class BstsPrediction:
    """
    Posterior predictive distribution produced by the call to Bsts.pred.
    """
    def __init__(self, distribution, original_series):
        """
        Create a BstsPrediction object.

        Args:

          distribution: A numpy array of either 2 or 3 dimensions.  In either
            case the first dimension represents MCMC draws and the last
            represents time points.  If three dimensions are present, the
            middle dimension represents the individual state components.

          original_series: The original time series of response values.
        """
        if len(distribution.shape) == 3:
            self.distribution = distribution[:, -1, :]
            self.components = distribution[:, :-1, :]
        else:
            self.distribution = distribution
            self.components = None
        self.posterior_mean = distribution.mean(axis=0)
        self._original_series = original_series

    @property
    def original_series(self):
        return self._original_series

    @property
    def posterior_predictive_distribution(self):
        """
        The posterior predictive distribution of new outcomes, including any
        observational noise.
        """
        return self.distribution

    @property
    def posterior_mean_distribution(self):
        """
        The distribution of the sum of the latent state contributions to the
        prediction.  If the model contains a link function, the prediction is
        on the link scale (e.g. it is the distribution of logit(p_t), not p_t).
        """
        if self.components is None:
            raise Exception("The BstsPrediction was not created with separate "
                            "components.")
        return self.components.sum(axis=1)

    def plot(self, original_series=True, ax=None, **kwargs):
        """
        Plot the predictive distribution on the supplied axes.

        Args:
          original_series: If True then plot the full original_series along
            with the prediction.  If False then only plot the prediction.  If
            an integer, then plot the last 'original_series' observations from
            the original series.
          ax: A plt.axes object on which to draw the plot.  If None, then a new
            figure and axes will be created.
          **kwargs: Extra arguments will be passed to
            'plot_dynamic_distribution'.

        Returns:
          The 'ax' object on which the plot is drawn.
        """
        horizon = self.distribution.shape[1]

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = None

        if (
                isinstance(self._original_series, pd.Series)
                and pd.api.types.is_datetime64_any_dtype(
                    self._original_series.index.dtype)
        ):
            plot_with_dates = True
            original_timestaps = pd.Series(self._original_series.index,
                                           dtype="datetime64[ns]")
            extended_timestamps = extend_timestamps(
                original_timestaps, num_steps=horizon)

        else:
            plot_with_dates = False
            start = len(self._original_series)
            end = start + self.distribution.shape[1]
            original_timestaps = np.arange(start)
            extended_timestamps = np.arange(start, end)

        if original_series is True:
            start = original_timestaps[0]
            plot_periods = len(self._original_series) + horizon
            original = R.to_numpy(self._original_series)
        elif isinstance(original_series, int):
            start = original_timestaps.iloc[-original_series]
            plot_periods = original_series + horizon
            original = self._original_series.values[-original_series:]
        elif original_series is False:
            start = extended_timestamps[0]
            plot_periods = horizon
            original = None
        else:
            raise Exception(
                "Unexpected value passed for 'original_series'.  Acceptable "
                "values are True, False, and int."
            )
        end = extended_timestamps[-1]

        if plot_with_dates:
            plotting_timestamps = pd.date_range(
                start=start, end=end, periods=plot_periods)
        else:
            plotting_timestamps = np.arange(start, end)

        if original is not None:
            ax.plot(plotting_timestamps[:len(original)],
                    original)

        R.plot_dynamic_distribution(
            self.distribution,
            extended_timestamps,
            ax=ax,
            **kwargs)

        ax.plot(extended_timestamps,
                self.posterior_mean,
                color="green")
        ax.plot(extended_timestamps,
                np.quantile(self.distribution, .025, axis=0),
                color="green", linestyle="dashed")
        ax.plot(extended_timestamps,
                np.quantile(self.distribution, .975, axis=0),
                color="green", linestyle="dashed")
        ax.set_xlim((plotting_timestamps[0], plotting_timestamps[-1]))

        return fig, ax


def extend_timestamps(timestamps, num_steps, dt: pd.Timedelta = None):
    """
    Extend a sequence of time stamps by a given number of steps.

    Args:
      timestamps: A collection of timestamps convertible to a pd.Series with
        dtype 'datetime64[ns]'
      num_steps: The integer number of time steps that 'timestamps' should be
        extended.
      dt: The increment between time steps.  If not given then dt is the
        smallest positive increment between the values of 'timestamps.'

    Returns:
      A pd.Series of time stamps starting one unit after the largest timestamp
      and extending for 'num_steps' beyond, in increments of 'dt'.
    """
    if dt is None:
        # Set 'dt' to the smallest positive time delta.
        timestamps = pd.Series(np.unique(np.sort(timestamps)),
                               dtype="datetime64[ns]")
        dt = timestamps.diff()[1:]
        dt = np.min(dt[dt > pd.Timedelta(0)])

    if dt < pd.Timedelta(7, "days"):
        start = np.max(timestamps) + dt
        end = start + (num_steps - 1) * dt
        return pd.date_range(start=start, end=end, periods=num_steps)

    elif dt >= pd.Timedelta(28, "days") and dt <= pd.Timedelta(31, "days"):
        return pd.date_range(start=np.max(timestamps),
                             freq="M",
                             periods=num_steps + 1)[1:]

    elif dt >= pd.Timedelta(90, "days") and dt <= pd.Timedelta(93, "days"):
        return pd.date_range(start=np.max(timestamps),
                             freq="3M",
                             periods=num_steps + 1)[1:]
    elif dt >= pd.Timedelta(365, "days") and dt <= pd.Timedelta(366, "days"):
        return pd.date_range(start=np.max(timestamps),
                             freq="Y",
                             periods=num_steps + 1)[1:]
    else:
        raise Exception(f"Not sure how to extend a time delta of {dt}.")


# ===========================================================================
# Bsts models support different model families for observation error (Gaussian,
# logit, student, Poisson).  The different families have different requirements
# for data formatting, produce different parameters, etc.  The
# ObservationModelManager supports the BOOM observation model held in each Bsts
# models's self._model.

class ObservationModelManager(ABC):
    """
    Manages making space for and recording MCMC draws for specific families of
    observation models.
    """

    @property
    def niter(self):
        """
        The number of MCMC iterations that have been run.
        """
        if hasattr(self, "_residual_sd"):
            return len(self._residual_sd)
        else:
            return 0

    @abstractmethod
    def allocate_space(self, niter: int, time_dimension: int):
        """
        Create space for 'niter' MCMC draws of the observation model parameters.
        This will include regression coefficients if the observation model has
        been assigned a formula.

        Args:
          niter:  The number of iterations worth of space to allocate.
          time_dimension: The number of time points in the training data.
        """

    @abstractmethod
    def record_draw(self, iteration: int, model):
        """
        Record the model parameters at the given iteration.

        Args:
          iteration: The iteration number of the draw to record.
          model: The BOOM state space model object from which to extract the
            draw.
        """

    @property
    def has_regression(self):
        """
        Returns True iff the model has a static regression component.  If it
        does, the model must allocate space for self._regression_contribution.
        in the allocate_space method.
        """

    @property
    def regression_contribution(self):
        return getattr(self, "_regression_contribution", None)

    def plot_regression_contribution(self, fig, gridspec, time, burn, ylim,
                                     **kwargs):
        """
        If the model has a regression component, plot it on the supplied figure.

        Args:
          fig:  The plt.Figure on which to draw the plot.
          gridspec: The plt.gridspec object describing where on the figure the
            plot should be drawn.
          time:  The timestamps to plot on the time axis.
          burn:  The number of iterations to discard as burn-in.
          ylim:  A pair giving the lower and upper limits on the y axis.
          kwargs:  Extra arguments passed to plot_dynamic_distribution.

        Returns:
          The axes object generated by 'figure' and 'gridspec', if the model
          has a regression component.  Otherwise, None is returned.
        """
        if not self.has_regression:
            return None

        ax = fig.add_subplot(gridspec)

        if (burn > 0):
            curves = self._regression_contribution[int(burn):, :]
        else:
            curves = self._regression_contribution

        R.plot_dynamic_distribution(
            curves=curves,
            timestamps=time,
            ax=ax,
            ylim=ylim,
            **kwargs)

    @abstractmethod
    def format_prediction_data(self, prediction_data, **kwargs):
        """
        Return a tuple containing the prediction data expected by the predict
        method.  The formatting is kept as a separate method for efficiency
        reasons.  The prediction data will be formatted once, while the predict
        method will be called with the same prediction data once per MCMC draw.

        Args:
          prediction_data: For regression models this is a data frame
            containing the predictor variables to be used in the prediction.
            All the variables mentioned in the model formula used to train the
            model must be present.  Their order need not be the same.

            For pure time series models this is an integer giving the number of
            time periods to forecast.

          **kwargs: Extra arguments that may be needed for Poisson or logit
            models.  Poisson models accept an 'exposure' argument.  Logit
            models accept a 'trials' argument.
        """

    @abstractmethod
    def predict(self, model, formatted_prediction_data, boom_final_state, rng,
                separate_components=False, **kwargs):
        """
        Simulate from the posterior predictive distribution.
        Args:
          model: A BOOM state space model object of the type expected by the
            concrete child class.
          formatted_prediction_data:
          boom_final_state: A boom.Vector containing the simulated state draw
            at the final time point in the training data.
          rng:  A boom.RNG random number generator to use in the simulation.
          separate_components: If True then separate the contributions of each
            state component in the output.  See the "Returns:" section for
            implications.
          **kwargs: Extra arguments expected by specific model families.
              Typical examples are 'trials' for logit models or 'exposure' for
              Poisson models.  It is an error to supply unhandled extra
              arguments.

        Returns:
          If separate_components is False then the return is a numpy matrix
          representing the posterior predictive distribution.  Rows are MCMC
          draws.  Columns are time points.  Otherwise the return is a numpy
          3-way array representing the posterior predictive distribution.  The
          first index is MCMC draws, the second is state components, and the
          thrird is time points.  If there is no link function, then summing
          over state components yields the same result as a call to 'predict'
          with the same seed, including observation error.  If the model
          includes a link function then the state contributions are on the
          scale of the link function, and no observation error is present.
        """


class StateSpaceModelFactory(ABC):
    @staticmethod
    def create(family, formula):
        family = R.unique_match(
            family.lower(),
            ["gaussian", "student", "binomial", "poisson"])
        from .gaussian import (
            GaussianStateSpaceModelFactory,
            StateSpaceRegressionModelFactory,
        )
        from .logit import StateSpaceLogitModelFactory
        from .poisson import StateSpacePoissonModelFactory
        from .student import StateSpaceStudentModelFactory

        if family == "gaussian" and formula is None:
            return GaussianStateSpaceModelFactory()
        elif family == "gaussian":
            return StateSpaceRegressionModelFactory(formula)
        elif family == "student":
            return StateSpaceStudentModelFactory(formula)
        elif family == "poisson":
            return StateSpacePoissonModelFactory(formula)
        elif family == "binomial":
            return StateSpaceLogitModelFactory(formula)
        else:
            raise Exception(f"Unrecognized family {family}.")

    @abstractmethod
    def create_model(self, prior, data, rng):
        """
        Create the BOOM model object.  The prior is assigned to the observation
        model, and the data is assigned to the model.  State will be assigned
        later.

        Args:
          prior: A prior distribution appropriate to the type of observation
            model.  Child classes will make and document specific assumptions
            about which priors are appropriate.
          data: The training data for the model.  In most cases this will be a
            pd.DataFrame (if the model contains a regression component) or a
            pd.Series otherwise.  The index of the data may contain timestamps.
          rng:  The boom random number generator.
        """

    @abstractmethod
    def create_observation_model_manager(self):
        """
        Return an ObservationModelManager object appropriate to the concrete
        observation model.
        """


def _find_state_contribution_ylim(state_models, burn):
    """
    Find the range of the values in the state contributions.
    Args:
      state_models:  List of state models.
      burn:  The number of MCMC iterations to discard as burn-in.

    Returns:
      A pair (lower, upper) giving the min and max values of state_contribution
      across models.
    """
    if (burn is None) or (burn < 0):
        burn = 0
    burn = int(burn)
    mins = [np.min(model._state_contribution[burn:, :])
            for model in state_models]
    maxs = [np.max(model._state_contribution[burn:, :])
            for model in state_models]
    return (np.min(mins), np.max(maxs))


def compare_bsts_models(models, burn=None, colors=None,
                        xlab="Time", ylab="Cumulative Absolute Error",
                        main="Model Comparison", fig=None, **kwargs):
    """
    Plot the cumulative absolute one-step prediction errors for a collection of
    bsts models.  Lower errors are good.  The plot helps you determine the time
    periods when poorly performing models accumulated the most error.

    Args:
      models: A collection of models.  If the colletion is a dict the dict keys
        will be used as labels in the plot.
      burn:  The number of MCMC iterations to be discarded as burn-in.
      colors:  The colors to use for the different models.
      xlab:  Label for the time dimension.
      main:  Main plot title.
      fig: A plt.Figure object on which to draw the plot.  If None then a
        Figure object will be created.
      **kwargs:  Extra arguments passed to the plotting functions.

    Returns:
      The 'fig' object on which the plot is drawn.
    """

    if not R.is_iterable(models):
        raise Exception("Expected a collection of models.")

    if not isinstance(models, dict):
        models = {f"Model {i+1}": model for i, model in enumerate(models)}

    model_names = list(models.keys())
    original_series = models[model_names[0]].original_series

    if fig is None:
        fig = plt.figure()

    if burn is None:
        burn = np.max([x.suggest_burn() for x in models.values()])

    num_models = len(models)
    if colors is None:
        colors = ["black", "red", "blue", "green"]
        colors = [x for x in colors for i in range(4)]
        colors = R.recycle(colors, num_models)

    line_styles = R.recycle(["-", "--", ":" "-."], num_models)

    gridspec = fig.add_gridspec(2, 1, hspace=0)
    bottom_panel = fig.add_subplot(gridspec[1, 0])
    bottom_panel.tick_params(bottom=True, right=True, left=False,
                             labelbottom=True, labelright=True, labelleft=False)
    bottom_panel.grid(linestyle=":", color="gray", linewidth=0.5)
    bottom_panel.set_ylabel("Original Series")

    top_panel = fig.add_subplot(gridspec[0, 0], sharex=bottom_panel)
    top_panel.tick_params(bottom=False, top=False, left=True, right=False,
                          labelbottom=False)
    top_panel.grid(linestyle=":", color="gray", linewidth=0.5)
    top_panel.set_title(main)
    top_panel.set_ylabel(ylab)

    R.plot_ts(original_series, ax=bottom_panel, xlab=xlab)
    counter = 0
    for model_name, model in models.items():
        errors = model.one_step_prediction_errors(burn=burn).mean(axis=0)
        cumulative_errors = pd.Series(
            np.cumsum(np.abs(errors)),
            index=original_series.index
        )
        top_panel.plot(cumulative_errors, color=colors[counter],
                       linestyle=line_styles[counter],
                       label=model_name, **kwargs)
        counter += 1

    top_panel.legend()

    return fig
