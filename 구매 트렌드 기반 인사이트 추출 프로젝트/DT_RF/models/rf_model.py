from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tree_model import TreeModel
from base_model import Classifier, Regressor

class RandomForestModel(TreeModel):
    def __init__(self, model_class, hyperparameters=None):
        super().__init__()
        default_params = {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
        if hyperparameters is not None:
            default_params.update(hyperparameters)
        self.model = self._create_model(model_class, default_params)

    def _create_model(self, model_class, hyperparameters):
        return model_class(**hyperparameters)

class RandomForestClassifierModel(RandomForestModel, Classifier):
    def __init__(self, hyperparameters=None):
        super().__init__(RandomForestClassifier, hyperparameters)
        Classifier.__init__(self)

class RandomForestRegressorModel(RandomForestModel, Regressor):
    def __init__(self, hyperparameters=None):
        super().__init__(RandomForestRegressor, hyperparameters)
        Regressor.__init__(self)
