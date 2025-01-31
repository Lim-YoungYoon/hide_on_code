import pandas as pd

class BaseDataLoader:
    def __init__(self):
        pass
        
    # CSV 파일을 읽어서 DataFrame으로 반환하는 메서드
    def load_data(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"데이터 로드 중 에러 발생: {e}")
            return None

