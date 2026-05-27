from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# 1. 데이터셋 로드
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

# 2. train/test 분할 (80% 학습, 20% 테스트)
# 기존 기본 모델과 같은 조건으로 비교하기 위해 동일하게 설정
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. GradientBoostingRegressor 모델 생성 및 학습
# n_estimators는 순차적으로 학습할 약한 모델의 개수를 의미한다.
# learning_rate는 각 모델의 예측 결과를 얼마나 반영할지 조절하는 값이다.
# max_depth는 각 트리의 깊이를 제한하여 과적합을 줄이기 위해 설정한다.
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)

gb_model.fit(X_train, y_train)

# 4. 예측
y_pred = gb_model.predict(X_test)

# 5. MSE 평가
mse = mean_squared_error(y_test, y_pred)

print("GradientBoostingRegressor MSE:")
print(mse)