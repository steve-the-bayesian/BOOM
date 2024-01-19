class FakeModel:

    def __init__(self, nfactors):
        self._nfactors = nfactors

    def __str__(self):
        return f"FakeModel with {self._nfactors} factors."
