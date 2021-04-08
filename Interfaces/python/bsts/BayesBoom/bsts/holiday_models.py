import numpy as np
import pandas as pd
import BayesBoom.boom as boom
import BayesBoom.R as R
from .state_models import StateModel
from .holiday import Holiday, HolidayFactory


class RegressionHolidayStateModel(StateModel):
    """
    Models holidays using contstant "dummy variable" effects, specific to each
    holiday.  This model is static, i.e. not dynamic, i.e. Valentine's Day is
    modeled as having the same effect each year.

    This state model can only be used with daily data.
    """
    def __init__(self,
                 y,
                 parent_model,
                 time0=None,
                 prior: R.NormalPrior = None,
                 sdy: float = None):
        """
        Args:
          y:  The time series being modeled.

          holiday_list: A list containing either strings of recognized holiday
            names, or Holiday objects.

          time0: Either None, or an object convertible to a pd.Timestamp.
            Gives the date of the first day in the time series 'y'.  If None
            then y must be a pd.Series (or equivalent) with an index containing
            this information in its first entry.

          prior:
        """
        self._time0 = self._validate_time0(time0, y)
        self._holidays = {}
        self._holiday_names = []
        if sdy is None:
            sdy = np.nanstd(y, ddof=1)
        self._prior = self._validate_prior(prior, sdy)
        self._parent_model = parent_model
        self._build_state_model()

    def __repr__(self):
        return f"RegressionHolidayStateModel({self._holiday_names})"

    @property
    def label(self):
        if len(self._holiday_names) <= 2:
            return f"{self._holiday_names}"
        else:
            return f"Holidays[{len(self._holiday_names)}]"

    def add_holiday(self, name: str, holiday: Holiday):
        # The list self._holiday_names is redundant, but it ensures the order
        # of the holiday elments, which is needed for python versions <= 3.6.
        self._holiday_names.append(name)
        self._holidays[name] = holiday

    @property
    def state_error_dimension(self):
        return 1

    @property
    def state_dimension(self):
        return 1

    def allocate_space(self, niter, time_dimension):
        self._holiday_patterns = {}
        for name, holiday in self._holidays.items():
            self._holiday_patterns[name] = np.empty(
                niter,
                holiday.maximum_window_width)

    def record_state(self, iteration, state_matrix):
        for i, name in enumerate(self._holiday_names):
            self._holiday_patterns[name][iteration, :] = (
                self._state_model.holiday_pattern(i).to_numpy()
            )

    def restore_state(self, iteration):
        for i, name in enumerate(self._holiday_names):
            (
                self._state_model.set_holiday_pattern(
                    i, R.to_boom_vector(
                        self._holiday_patterns[name][iteration, :]))
            )

    def _validate_time0(self, time0, y):
        if time0 is None:
            time0 = y.index[0]
        return pd.Timestamp(time0)

    def _validate_prior(self, prior, sdy):
        if prior is None:
            prior = R.NormalPrior(0, sdy)
        if not isinstance(prior, R.NormalPrior):
            raise Exception("Expected a prior of type R.NormalPrior.")
        return prior

    def _build_state_model(self):
        self._state_model = boom.ScalarRegressionHolidayStateModel(
            R.to_boom_date(self._time0),
            self._parent_model,
            self._prior.boom())

        holiday_factory = HolidayFactory()
        for holiday in self._holiday_list:
            self._state_model.add_holiday(
                holiday_factory.create_boom_holiday(holiday))

    def __getstate__(self):
        payload = self.__dict__.copy()
        del payload["_state_model"]
        return payload

    def __setstate__(self, payload):
        self.__dict__ = payload
        self._build_state_model()
