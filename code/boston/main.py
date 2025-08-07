from boston_loader import BostonDataLoader
from boston_preprocessor import BostonPreprocessor
from DT import DecisionTreeModel
from RF import RandomForestModel
from XGB import XGBModel

def main():
    # 1) 데이터 로드
    data_loader = BostonDataLoader("boston.csv")
    df = data_loader.load_data()

    # 2) 전처리
    preprocessor = BostonPreprocessor(df)
    print("=== 결측치 확인 ===")
    print(preprocessor.check_missing())

    preprocessor.handle_missing()
    print("\n=== 결측치 처리 후 ===")
    print(preprocessor.check_missing())

    X_train, X_test, y_train, y_test = preprocessor.scale_split(
        target='PRICE', 
        test_size=0.2, 
        random_state=42
    )

    # 3) 모델 학습 및 평가

    ## 3.1 Decision Tree
    dt_model = DecisionTreeModel()
    dt_model.train(X_train, y_train)
    dt_mse, dt_r2 = dt_model.evaluate(X_test, y_test)
    print("\n[Decision Tree] MSE:", dt_mse, "R2:", dt_r2)

    ## 3.2 Random Forest
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    rf_mse, rf_r2 = rf_model.evaluate(X_test, y_test)
    print("[Random Forest] MSE:", rf_mse, "R2:", rf_r2)

    ## 3.3 XGBoost
    xgb_model = XGBModel(n_estimators=100, random_state=42)
    xgb_model.train(X_train, y_train)
    xgb_mse, xgb_r2 = xgb_model.evaluate(X_test, y_test)
    print("[XGBoost] MSE:", xgb_mse, "R2:", xgb_r2)

if __name__ == "__main__":
    main()
