from .BaseModels import DoubleModel


class BetaModel(DoubleModel):
    """Beta distribution, typically used as a prior over a probability."""

    def __init__(self, a=1.0, b=1.0):
        self._a = float(a)
        self._b = float(b)
        self._boom_model = None

    @property
    def mean(self):
        return self._a / (self._a + self._b)

    def boom(self):
        if self._boom_model is not None:
            return self._boom_model
        import BayesBoom.boom as boom
        self._boom_model = boom.BetaModel(self._a, self._b)
        return self._boom_model

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __getstate__(self):
        payload = dict(self.__dict__)
        payload["_boom_model"] = self._boom_model is not None
        return payload

    def __setstate__(self, payload):
        self.__dict__ = payload
        if payload["_boom_model"]:
            self._boom_model = None
            self.boom()
