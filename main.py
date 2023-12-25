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

# 결측치 처리
for name, df in dataframes.items():
    df.fillna(0, inplace=True)
import pandas as pd

# 'sales.csv' 파일 불러오기
sales_df = pd.read_csv("data/sales.csv")

# 데이터 확인
# print(sales_df.head())

# 6번째 열부터 날짜 데이터로 가정하고, melt 함수를 사용
sales_melted = sales_df.melt(id_vars=sales_df.columns[:6],  # 첫 5개 열을 id_vars로 유지
                             var_name='date',
                             value_name='sales')

# 'date' 열을 datetime 객체로 변환
sales_melted['date'] = pd.to_datetime(sales_melted['date'])

# 데이터 확인
print(sales_melted.head())

# 결측치를 0 또는 평균값 등으로 대체 (여기서는 0으로 대체)
sales_melted.fillna(0, inplace=True)

# 시계열 데이터 시각화
sales_melted['sales'].plot(figsize=(15, 6))
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
# plt.show()

# 예시: 'sales' 데이터의 일부 시계열 데이터 시각화
plt.figure(figsize=(15, 6))
sns.lineplot(data=dataframes['sales'].iloc[:, 6:16])
plt.title('Sales Time Series for First Few Dates')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
# plt.show()


keyword_df = pd.read_csv("data/brand_keyword_cnt.csv")
sales_df = pd.read_csv("data/sales.csv")

# 키워드 데이터의 날짜 열을 행으로 변환
# 예시 코드에서 id_vars는 첫 5개 열을 가정하고 있습니다.

keyword_melted = keyword_df.melt(id_vars=['브랜드'], var_name='date', value_name='keyword_count')
keyword_melted['date'] = pd.to_datetime(keyword_melted['date'])

# 판매 데이터의 날짜 열을 행으로 변환
sales_melted = sales_df.melt(id_vars=sales_df.columns[:6],  # 첫 5개 열을 id_vars로 유지
                             var_name='date',
                             value_name='sales')

# 'date' 열을 datetime 객체로 변환
sales_melted['date'] = pd.to_datetime(sales_melted['date'])

# 데이터 결합
combined_df = pd.merge(keyword_melted, sales_melted, on=['브랜드', 'date'])

# 피어슨 상관 계수 계산
correlation = combined_df[['keyword_count', 'sales']].corr()
print(correlation)


# 상관 관계 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Between Keyword Count and Sales')
plt.show()
