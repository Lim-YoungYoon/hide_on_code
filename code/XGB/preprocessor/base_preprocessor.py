import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class BasePreProcessor:
    def __init__(self, df, target_col):
        self.scaler = StandardScaler()
        self.df = df
        self.target_col = target_col

    def run(self):
        print("전처리 시작")
        self.encode_categorical_features()
        self.handle_missing_values()
        self.scale_numeric_features()
        print("전처리 완료")
        
    def split_data(self, test_size=0.2, random_state=42):
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def scale_numeric_features(self):
        numeric_cols = self.df.drop(self.target_col, axis=1).select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])

    def handle_missing_values(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def encode_categorical_features(self):
        raise NotImplementedError("Subclasses should implement this method.")
