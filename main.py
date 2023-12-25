import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# 데이터 불러고이
sales_data_path = 'data/sales.csv'
sales_data = pd.read_csv(sales_data_path)

# 데이터 구조
print(sales_data.head())

# 데이터를 시계열 형식으로 변환, 매일의 상품의 판매 데이터를 집계
# 시계열의 날짜 열 추출
date_columns = sales_data.columns[6:]  # 날짜열은 7번째 부터 시작

# 모든 제품의 일별 판매량을 더함
daily_sales = sales_data[date_columns].sum()

# 시계열 데이터프레임 만들기
time_series_data = pd.DataFrame({'Date': pd.to_datetime(date_columns), 'Total Sales': daily_sales.values})

# 시계열 데이터의 행 확인
print(time_series_data.head())

sns.set(style="whitegrid")

# 시계열 데이터 시각화
plt.figure(figsize=(15, 6))
plt.plot(time_series_data['Date'], time_series_data['Total Sales'], label='Total Sales')
plt.title('Daily Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
# plt.show()

# Date 열을 인덱스로 설정
time_series_data.set_index('Date', inplace=True)

# Total Sales 열을 선택해서 계절성 분해 수행
# additive 모델을 사용, 주기는 30일
decomposition = seasonal_decompose(time_series_data['Total Sales'], model='additive', period=30)

# 시각화를 위한 플롯 사이즈 설정
plt.figure(figsize=(14, 8))

# 원본 시계열 데이터
plt.subplot(411)
plt.plot(decomposition.observed, label='Original')
plt.legend(loc='upper left')
plt.title('Original Time Series')

# 추세 성분
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.title('Trend Component')

# 계절성 성분
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.title('Seasonal Component')

# 잔차 성분
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual')
plt.legend(loc='upper left')
plt.title('Residual Component')

plt.tight_layout()
plt.show()
