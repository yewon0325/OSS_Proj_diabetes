from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 1. 데이터셋 로드
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

# 2. train/test 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. 다중 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 예측
y_pred = model.predict(X_test)

# 5. MSE 평가
mse = mean_squared_error(y_test, y_pred)

print("MSE:")
print(mse)

# 6. 실제값과 예측값 비교 그래프
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)

# 실제값 = 예측값 기준선
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("실제값")
plt.ylabel("예측값")
plt.title("실제값과 예측값 비교")

# 그래프 저장 폴더 생성
os.makedirs("figures", exist_ok=True)

# 그래프 저장
plt.savefig("figures/실제값과 예측값 비교.png", dpi=300, bbox_inches="tight")

# 그래프 화면 출력
plt.show()

# 7. 테스트 데이터 일부 예측 결과 확인
print("\n예측 결과 확인:")
print("실제값\t예측값")

for i in range(10):
    print(f"{y_test[i]:.1f}\t{y_pred[i]:.1f}")