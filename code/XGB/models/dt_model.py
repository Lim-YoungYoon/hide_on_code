from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tree_model import TreeModel
from base_model import Classifier, Regressor

class DecisionTreeModel(TreeModel):
    """
    tree_model의 TreeModel을 상속받아
    디시젼트리 모델을 구현한 베이스 클래스입니다.
    """
    def __init__(self, model_class, hyperparameters=None):
        super().__init__()
        default_params = {'max_depth': None, 'random_state': 42}
        if hyperparameters is not None:
            default_params.update(hyperparameters)
        self.model = self._create_model(model_class, default_params)

    def _create_model(self, model_class, hyperparameters):
        return model_class(**hyperparameters)

class DecisionTreeClassifierModel(DecisionTreeModel, Classifier):
    """
    DecisionTreeModel과 base_model의 Classifier를 상속받아
    디시젼트리 분류 모델을 구현한 클래스입니다.
    """
    def __init__(self, hyperparameters=None):
        super().__init__(DecisionTreeClassifier, hyperparameters)
        Classifier.__init__(self)

class DecisionTreeRegressorModel(DecisionTreeModel, Regressor):
    """
    DecisionTreeModel과 base_model의 Regressor를 상속받아
    디시젼트리 회귀 모델을 구현한 클래스입니다.
    """
    def __init__(self, hyperparameters=None):
        super().__init__(DecisionTreeRegressor, hyperparameters)
        Regressor.__init__(self)