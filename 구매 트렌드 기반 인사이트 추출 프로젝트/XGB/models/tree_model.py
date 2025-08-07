import pandas as pd
import matplotlib.pyplot as plt
from base_model import BaseModel
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class TreeModel(BaseModel):
    def __init__(self):
        super().__init__()

    def _create_model(self, model_class, hyperparameters):
        raise NotImplementedError("이 메서드는 자식 클래스에서 구현해야 합니다.")

    def get_feature_importances_df(self, feature_names):
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)
        return fi_df
    
    def plot_feature_importances(self, feature_names, top_k=None):
        fi_df = self.get_feature_importances_df(feature_names)
        if top_k is not None:
            fi_df = fi_df.head(top_k)

        # 특징 중요도를 수평 막대 그래프로 시각화
        plt.barh(fi_df['feature'], fi_df['importance'])
        
        # y축을 반전시켜 가장 중요한 특징이 위에 오도록 설정
        plt.gca().invert_yaxis()
        
        # 그래프 제목 설정, top_k가 주어지면 상위 k개의 특징 중요도를 표시
        plt.title(f"Top {top_k} Feature Importances" if top_k is not None else "Feature Importances")
        
        # x축과 y축 레이블 설정
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        
        # 그래프를 화면에 표시
        plt.show()
