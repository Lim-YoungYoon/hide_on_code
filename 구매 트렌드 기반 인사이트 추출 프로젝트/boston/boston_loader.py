import pandas as pd

class BostonDataLoader:
    """
    보스턴 집값 CSV 파일을 로드하는 클래스
    """
    def __init__(self, csv_path: str):
        """
        Parameters
        ----------
        csv_path : str
            보스턴 집값 CSV 파일 경로
        """
        self.csv_path = csv_path

    def load_data(self) -> pd.DataFrame:
        """
        CSV 파일에서 데이터를 로드하여 pandas DataFrame 형태로 반환

        Returns
        -------
        df : pd.DataFrame
        """
        df = pd.read_csv(self.csv_path)
        return df
