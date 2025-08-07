from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class RandomForestModel:
    """
    Random Forest Regression 모델 클래스
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : dict
            RandomForestRegressor에 전달할 파라미터
        """
        self.model = RandomForestRegressor(**kwargs)
    
    def train(self, X_train, y_train):
        """
        모델 학습
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """
        예측 값 반환
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        성능 평가 (MSE, R2)
        
        Returns
        -------
        mse : float
            평균제곱오차
        r2 : float
            결정계수
        """
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2
