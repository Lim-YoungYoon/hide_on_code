from base_preprocessor import BasePreProcessor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class LoanDefaultDataPreprocessor(BasePreProcessor):
    def __init__(self, df, target_col):
        super().__init__(df, target_col)

    def handle_missing_values(self):
        # LoanDefaultDataPreprocessor에 특화된 결측치 처리 로직
        columns_to_drop = self.df.isna().sum()[self.df.isna().sum() < 1000].index.tolist()
        self.df = self.df.dropna(subset=columns_to_drop)
        cols_to_fill = {
            'dtir1': 'loan_amount',
            'property_value': 'loan_amount',
            'LTV': 'loan_amount',
            'income': 'Credit_Score',
            'loan_limit': 'loan_amount',
            'rate_of_interest': 'loan_amount',
            'Interest_rate_spread': 'loan_amount',
            'Upfront_charges': 'loan_amount'
        }
        for target_col, group_col in cols_to_fill.items():
            if self.df[target_col].dtype == 'object':
                mode_value = self.df[target_col].mode()[0]
                self.df[target_col] = self.df[target_col].fillna(mode_value)
            else:
                self._fill_missing_by_group(target_col, group_col)

    def encode_categorical_features(self):
        age_order = ['<25', '25-34', '35-44', '45-54', '55-64', '65-74', '>74']
        age_encoder = {age: i for i, age in enumerate(age_order)}
        if 'age' in self.df.columns:
            self.df['age'] = self.df['age'].map(age_encoder)
        label_encoding_columns = [
            'approv_in_adv', 'Credit_Worthiness', 'business_or_commercial',
            'Neg_ammortization', 'interest_only', 'lump_sum_payment',
            'loan_limit', 'open_credit', 'total_units',
            'submission_of_application', 'term'
        ]
        onehot_encoding_columns = [
            'Gender', 'loan_type', 'loan_purpose', 'construction_type',
            'occupancy_type', 'Secured_by', 'credit_type',
            'co-applicant_credit_type', 'Region', 'Security_Type'
        ]
        le = LabelEncoder()
        for col in label_encoding_columns:
            if col in self.df.columns:
                self.df[col] = le.fit_transform(self.df[col])
        for col in onehot_encoding_columns:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)

    def _fill_missing_by_group(self, target_col, group_col, n_bins=5):
        bin_col = f"{group_col}_bin"
        self.df[bin_col] = pd.qcut(self.df[group_col], q=n_bins, duplicates='drop')
        medians = self.df.groupby(bin_col)[target_col].transform('median')
        self.df[target_col] = self.df[target_col].fillna(medians)
        self.df.drop(bin_col, axis=1, inplace=True)
