import numpy as np

class DirichletModel:
    """Dirichlet prior over discrete probability distributions."""

    def __init__(self, counts):
        counts = np.array(counts)
        if not np.all(counts > 0):
            raise Exception("All elements of 'counts' must be positive.")
        self._counts = counts
        self._boom_model = None

    def boom(self):
        if self._boom_model is None:
            import BayesBoom.boom as boom
            self._boom_model = boom.DirichletModel(boom.Vector(self._counts))
        return self._boom_model

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __getstate__(self):
        ans = dict(self.__dict__)
        ans["_boom_model"] = self._boom_model is not None
        return ans

    def __setstate__(self, payload):
        self.__dict__ = payload
        if payload["_boom_model"]:
            self._boom_model = None
            self.boom()


DirichletPrior = DirichletModel
