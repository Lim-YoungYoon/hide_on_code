from base_data_loader import BaseDataLoader

class LoanDefaultDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()
        
    def load_data(self, csv_path='Loan_Default.csv'):
        # 부모 클래스의 load_data 메서드를 사용하여 데이터 로드
        df = super().load_data(csv_path)
        
        return df
