from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
class BaseModel:
    """
    베이스 모델 클래스
    공통적으로 사용하는 구조를 정의하고,
    분류(Classifier), 회귀(Regressor)에서 상속받아 사용합니다.
    """
    def __init__(self):
        self.metrics = {}

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        raise NotImplementedError("이 메서드는 자식 클래스에서 구현해야 합니다.")

class Classifier(BaseModel):
    def __init__(self):
        super().__init__()

    def evaluate(self, y_true, y_pred):
        # average 파라미터는 분류 문제가 이진인지 멀티클래스인지에 따라 다르게 설정됩니다.
        #   - average='binary': 양성과 음성을 구분하는 이진 분류에서 주로 사용됩니다.
        #   - average='macro': 각 클래스별 지표를 계산한 뒤 단순 평균하여 전체 성능을 측정합니다.
        # 여기서는 y_true에 포함된 클래스의 개수를 확인하여, 이진 분류인지 멀티클래스 분류인지 판단합니다.

        unique_classes = len(set(y_true))
        average = 'binary' if unique_classes == 2 else 'macro'
        self.metrics['acc'] = accuracy_score(y_true, y_pred)
        self.metrics['recall'] = recall_score(y_true, y_pred, average=average)
        self.metrics['precision'] = precision_score(y_true, y_pred, average=average)
        self.metrics['f1'] = f1_score(y_true, y_pred, average=average)
        return self.metrics

class Regressor(BaseModel):
    def __init__(self):
        super().__init__()

    def evaluate(self, y_true, y_pred):
        self.metrics['mae'] = mean_absolute_error(y_true, y_pred)
        self.metrics['mse'] = mean_squared_error(y_true, y_pred)
        self.metrics['r2score'] = r2_score(y_true, y_pred)
        return self.metrics
