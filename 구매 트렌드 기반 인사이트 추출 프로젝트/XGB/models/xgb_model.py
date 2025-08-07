# xgb_model.py
import xgboost as xgb
from tree_model import TreeModel
from base_model import Classifier

class XGBModel(TreeModel):
    def __init__(self, model_class, params=None):
        super().__init__()
        default_params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        if params is not None:
            default_params.update(params)
        self.model = self._create_model(model_class, default_params)

    def _create_model(self, model_class, params):
        return model_class(**params)

class XGBClassifierModel(XGBModel, Classifier):
    def __init__(self, params=None):
        super().__init__(xgb.XGBClassifier, params)
        Classifier.__init__(self)