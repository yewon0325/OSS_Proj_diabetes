from sklearn.datasets import load_diabetes
import pandas as pd

# 데이터셋 로드
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

# DataFrame으로 변환
df = pd.DataFrame(X, columns=diabetes.feature_names)
df["target"] = y

print("데이터 크기:")
print(df.shape)

print("\n특성 이름:")
print(diabetes.feature_names)

print("\n데이터 상위 5개:")
print(df.head())
