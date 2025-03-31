import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from .boom_py_utils import to_boom_vector, to_boom_matrix

class DataBuilder(ABC):
    """
    A generic utility for converting Python data types to BOOM data.
    """
    @abstractmethod
    def build_boom_data(self, data):
        """
        """


class IntDataBuilder(DataBuilder):
    def build_boom_data(self, data):
        import BayesBoom.boom as boom
        return [boom.IntData(int(x)) for x in data]


class DoubleDataBuilder(DataBuilder):
    def build_boom_data(self, data):
        import BayesBoom.boom as boom
        return [boom.DoubleData(float(x)) for x in data]


class VectorDataBuilder(DataBuilder):
    def build_boom_data(self, data):
        import BayesBoom.boom as boom
        if isinstance(data, np.ndarray):
            return [boom.VectorData(to_boom_vector(data[i, :]))
                    for i in range(data.shape[0])]
        elif isinstance(data, pd.DataFrame):
            return [boom.Vector(to_boom_vector(data.iloc[i, :]))
                    for i in range(data.shape[0])]
        else:
            raise Exception("VectorDataBuilder could not build a BOOM data "
                            "set with these inputs.")


class LabelledCategoricalDataBuilder(DataBuilder):
    def __init__(self, categories):
        self._categories = categories
        import BayesBoom.boom as boom
        self._boom_category_key = boom.CatKey([str(x) for x in categories])

    def build_boom_data(self, data):
        import BayesBoom.boom as boom
        return [boom.CategoricalData(str(x), self._boom_category_key)
                for x in data]

class UnlabelledCategoricalDataBuilder(DataBuilder):
    def __init__(self, nlevels):
        import BayesBoom.boom as boom
        self._boom_category_key = boom.FixedSizeIntCatKey(int(nlevels))

    def build_boom_data(self, data):
        import BayesBoom.boom as boom
        return [boom.CategoricalData(int(x), self._boom_category_key)
                for x in data]


class LabelledMarkovDataBuilder(DataBuilder):
    def __init(self, categories):
        import BayesBoom.boom as boom
        self._categories = categories
        self._boom_category_key = boom.CatKey([str(x) for x in categories])

    def build_boom_data(self, data):
        import BayesBoom.boom as boom
        ans = []
        for i in range(len(data)):
            if i == 0:
                ans.append(boom.MarkovData(data[i], self._boom_category_key))
            else:
                ans.append(boom.MarkovData(data[i], ans[i-1]))
        return ans


class MarkovSufDataBuilder(DataBuilder):
    def __init__(self):
        pass

    def build_boom_data(self, data):
        return [x.boom() for x in data]


class UnlabelledMarkovDataBuilder(DataBuilder):
    def __init__(self, nlevels):
        import BayesBoom.boom as boom
        self._boom_category_key = boom.FixedSizeIntCatKey(int(nlevels))

    def build_boom_data(self, data):
        import BayesBoom.boom as boom
        ans = []
        for i in range(len(data)):
            if i == 0:
                ans.append(boom.MarkovData(data[i], self._boom_category_key))
            else:
                ans.append(boom.MarkovData(data[i], ans[i-1]))
        return ans


class MultilevelCategoricalDataBuilder(DataBuilder):
    def __init__(self, boom_taxonomy, sep="/"):
        self._boom_taxonomy = boom_taxonomy
        self._sep = sep

    def build_boom_data(self, data):
        import BayesBoom.boom as boom
        ans = [
            boom.MultilevelCategoricalData(self._boom_taxonomy, x, self._sep)
            for x in data
        ]
        return ans;
