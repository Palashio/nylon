import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class SklearnClassifier(BentoService):

    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        return self.artifacts.model.predict(df)


