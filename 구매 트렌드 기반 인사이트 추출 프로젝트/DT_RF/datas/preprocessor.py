import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, df, target_col):
        self.scaler = StandardScaler()
        self.df = df
        self.target_col = target_col

    def run(self):
        print("전처리 시작")
        print("\t1. 카테고리컬 데이터를 인코딩 합니다.")
        self.encode_categorical_features()

        print("\t2. 결측치를 처리 합니다.")
        self.handle_missing_values()
        
        print("\t3. 수치형 데이터를 스케일링 합니다.")
        self.scale_numeric_features()
        
        print("전처리 완료")
        

    def split_data(self, test_size=0.2, random_state=42):
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    
    def scale_numeric_features(self):
        """
        수치형 변수 스케일링 메서드
        """
        numeric_cols = self.df.drop(self.target_col, axis=1).select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])

    def _fill_missing_by_group(self, target_col, group_col, n_bins=5):
        """
        구간별 결측치 채우기 메서드
        """
        bin_col = f"{group_col}_bin"
        self.df[bin_col] = pd.qcut(self.df[group_col], q=n_bins, duplicates='drop')
        
        # 각 구간별 중앙값 계산
        medians = self.df.groupby(bin_col)[target_col].transform('median')
        
        # 결측치 채우기
        self.df[target_col] = self.df[target_col].fillna(medians)
        
        self.df.drop(bin_col, axis=1, inplace=True)

    def handle_missing_values(self):
        """
        결측치 처리 메서드
        """
        # 결측치가 1000개 미만인 컬럼 추출
        columns_to_drop = self.df.isna().sum()[self.df.isna().sum() < 1000].index.tolist()
        
        # 해당 컬럼들의 결측치가 있는 행 제거
        self.df = self.df.dropna(subset=columns_to_drop)
        
        # 결측치 처리가 필요한 컬럼들과 그룹화 기준 컬럼을 매핑
        cols_to_fill = {
            'dtir1': 'loan_amount',  # 대출금액과 관련이 있을 것으로 예상
            'property_value': 'loan_amount',  # 대출금액과 부동산 가치는 연관성이 높음
            'LTV': 'loan_amount',  # LTV는 대출금액과 직접적인 관계
            'income': 'Credit_Score',  # 소득과 신용점수는 상관관계가 있음
            'loan_limit': 'loan_amount',  # 대출한도와 대출금액은 밀접한 관계
            'rate_of_interest': 'loan_amount',  # 대출금액과 관련이 있을 것으로 예상
            'Interest_rate_spread': 'loan_amount',  # 대출금액과 관련이 있을 것으로 예상
            'Upfront_charges': 'loan_amount'  # 대출금액과 관련이 있을 것으로 예상
        }

        # 각 컬럼에 대해 결측치 처리 수행
        for target_col, group_col in cols_to_fill.items():
            # 문자열 데이터가 있는지 확인
            if self.df[target_col].dtype == 'object':
                # 문자열 데이터는 최빈값으로 대체
                mode_value = self.df[target_col].mode()[0]
                self.df[target_col] = self.df[target_col].fillna(mode_value)
            else:
                # 숫자형 데이터는 그룹별 중앙값으로 대체
                self._fill_missing_by_group(target_col, group_col)


    def encode_categorical_features(self):
        """범주형 변수 인코딩 메서드"""
        
        # 나이 컬럼 순서에 맞춰 라벨 인코딩
        age_order = ['<25', '25-34', '35-44', '45-54', '55-64', '65-74', '>74']
        age_encoder = {age: i for i, age in enumerate(age_order)}
        if 'age' in self.df.columns:
            self.df['age'] = self.df['age'].map(age_encoder)
            
        # 라벨 인코딩할 컬럼들 (순서가 있거나 이진분류인 경우)
        label_encoding_columns = [
            'approv_in_adv',     # 사전승인 여부 - 이진분류
            'Credit_Worthiness', # 신용도 - 순서가 있음
            'business_or_commercial', # 사업/상업용 여부 - 이진분류
            'Neg_ammortization', # 음의 상각 여부 - 이진분류
            'interest_only',     # 이자만 납부 여부 - 이진분류
            'lump_sum_payment',  # 일시불 납부 여부 - 이진분류
            'loan_limit',        # 대출 한도 - 순서가 있음
            'open_credit',       # 신용 개설 여부 - 이진분류
            'total_units',       # 총 유닛 수 - 순서가 있음
            'submission_of_application', # 신청서 제출 방식 - 이진분류
            'term',              # 대출 기간 - 순서가 있음
        ]

        # 원핫 인코딩할 컬럼들 (순서가 없는 범주형 변수)
        onehot_encoding_columns = [
            'Gender',            # 성별 - 순서 없는 범주
            'loan_type',         # 대출 유형 - 순서 없는 범주
            'loan_purpose',      # 대출 목적 - 순서 없는 범주
            'construction_type', # 건설 유형 - 순서 없는 범주
            'occupancy_type',    # 점유 유형 - 순서 없는 범주
            'Secured_by',        # 담보 유형 - 순서 없는 범주
            'credit_type',       # 신용 유형 - 순서 없는 범주
            'co-applicant_credit_type', # 공동신청자 신용유형 - 순서 없는 범주
            'Region',            # 지역 - 순서 없는 범주
            'Security_Type'      # 보안 유형 - 순서 없는 범주
        ]

        # 라벨 인코딩 적용
        le = LabelEncoder()
        for col in label_encoding_columns:
            if col in self.df.columns:
                self.df[col] = le.fit_transform(self.df[col])

        # 원핫 인코딩 적용
        for col in onehot_encoding_columns:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
