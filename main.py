import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 파일 경로
file_paths = {
    "brand_keyword_cnt": "data/brand_keyword_cnt.csv",
    "product_info": "data/product_info.csv",
    "sales": "data/sales.csv",
    "sample_submission": "data/sample_submission.csv",
    "train": "data/train.csv"
}

# 데이터프레임으로 불러오기
dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

# 각 데이터프레임의 첫 5행 확인
for name, df in dataframes.items():
    print(f"---{name}---")
    print(df.head())
    print("\n")

# 결측치 확인
for name, df in dataframes.items():
    print(f"---{name} 결측치---")
    print(df.isnull().sum())
    print("\n")

# 기술 통계 분석
for name, df in dataframes.items():
    # 수치형 데이터에 대한 기술 통계만 계산
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        print(f"---{name} 기술 통계---")
        print(df.describe())
        print("\n")

# 예시: 'sales' 데이터의 일부 시계열 데이터 시각화
plt.figure(figsize=(15, 6))
sns.lineplot(data=dataframes['sales'].iloc[:, 6:16])
plt.title('Sales Time Series for First Few Dates')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()
