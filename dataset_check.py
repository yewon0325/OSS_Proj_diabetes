from sklearn.datasets import load_diabetes
import numpy as np

# 데이터셋 로드
diabetes = load_diabetes()

# target값 확인
y = diabetes.target

print("target 예시:")
print(y[:20])

print("\ntarget 고유값 개수:")
print(len(np.unique(y)))