import numpy as np
import pandas as pd
import BayesBoom.boom as boom
from abc import ABC, abstractmethod
from .boom_py_utils import to_boom_vector

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
        return [boom.IntData(int(x)) for x in data]

    
class DoubleDataBuilder(DataBuilder):
    def build_boom_data(self, data):
        return [boom.DoubleData(float(x)) for x in data]

    
class VectorDataBuilder(DataBuilder):
    def build_boom_data(self, data):
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
        self._boom_category_key = boom.CatKey([str(x) for x in categories])
        
    def build_boom_data(self, data):
        return [boom.CategoricalData(str(x), self._boom_category_key)
                for x in data]
        
class UnlabelledCategoricalDataBuilder(DataBuilder):
    def __init__(self, nlevels):
        self._boom_category_key = boom.FixedSizeIntCatKey(int(nlevels))
        
    def build_boom_data(self, data):
        return [boom.CategoricalData(int(x), self._boom_category_key)
                for x in data]

    
class LabelledMarkovDataBuilder(DataBuilder):
    def __init(self, categories):
        self._categories = categories
        self._boom_category_key = boom.CatKey([str(x) for x in categories])

    def build_boom_data(self, data):
        ans = []
        for i in range(len(data)):
            if i == 0:
                ans.append(boom.MarkovData(data[i], self._boom_category_key))
            else:
                ans.append(boom.MarkovData(data[i], ans[i-1]))
        return ans

                
class UnlabelledMarkovDataBuilder(DataBuilder):
    def __init__(self, nlevels):
        self._boom_category_key = boom.FixedSizeIntCatKey(int(nlevels))
        
    def build_boom_data(self, data):
        ans = []
        for i in range(len(data)):
            if i == 0:
                ans.append(boom.MarkovData(data[i], self._boom_category_key))
            else:
                ans.append(boom.MarkovData(data[i], ans[i-1]))
        return ans
