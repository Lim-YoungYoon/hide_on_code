import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BostonPreprocessor:
    """
    보스턴 집값 데이터 전처리를 담당하는 클래스
    """
    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            보스턴 집값 데이터
        """
        self.df = df.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def check_missing(self) -> pd.Series:
        """
        결측치 존재 여부 확인

        Returns
        -------
        pd.Series
            각 열별 결측치 개수
        """
        return self.df.isnull().sum()
    
    def handle_missing(self):
        """
        결측치 처리 (예시는 단순 dropna)
        필요에 따라 다른 방식(평균 대치 등)을 사용해도 무방합니다.
        """
        self.df.dropna(inplace=True)

    def scale_split(
        self, 
        target: str = 'MEDV', 
        test_size: float = 0.2, 
        random_state: int = 42
    ):
        """
        StandardScaler를 사용하여 피처(독립 변수)만 정규화한 뒤
        train, test 데이터로 분할

        Parameters
        ----------
        target : str, default='MEDV'
            종속 변수(집값 컬럼) 이름
        test_size : float, default=0.2
            테스트 세트 비율
        random_state : int, default=42
            랜덤 시드 값
        
        Returns
        -------
        X_train, X_test, y_train, y_test : np.ndarray
        """
        X = self.df.drop(columns=[target])
        y = self.df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, 
            y, 
            test_size=test_size, 
            random_state=random_state
        )

        return self.X_train, self.X_test, self.y_train, self.y_test
