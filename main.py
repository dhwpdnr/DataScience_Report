import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 불러오기
sales_data_path = 'data/sales.csv'
brand_keyword_data_path = 'data/brand_keyword_cnt.csv'
sales_data = pd.read_csv(sales_data_path)
brand_keyword_data = pd.read_csv(brand_keyword_data_path)

# sales_data에 대한 기본적인 통계 정보
sales_describe = sales_data.describe()

# sales_data의 결측값 확인
sales_missing_values = sales_data.isna().sum()

# 판매 데이터 중 일부 일자에 대한 판매 금액 분포 시각화
sample_columns = sales_data.columns[-5:]  # 마지막 5개 일자 선택
plt.figure(figsize=(15, 7))
for i, column in enumerate(sample_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=sales_data[column])
    plt.title(f'Distribution of Sales on {column}')
    plt.ylabel('Sales Amount')
plt.tight_layout()

# 시각화 결과 출력
sales_describe, sales_missing_values, plt.show()

# 모든 제품의 일별 총합
sales_daily_total = sales_data.iloc[:, 6:].sum()

# 날짜를 인덱스로 하는 시계열 데이터 생성
sales_time_series = pd.Series(sales_daily_total.values, index=pd.to_datetime(sales_daily_total.index))

# 시계열 데이터의 결측치 확인 및 처리
sales_time_series = sales_time_series.replace(0, np.nan)  # 0값을 NaN으로 대체
sales_time_series = sales_time_series.dropna()  # 결측치 제거

# 시계열 분해를 통한 추세, 계절성, 잔차 확인
decomposed = seasonal_decompose(sales_time_series, model='additive', period=30)  # 30일 주기로 계절성 분해

# 시계열 분해 결과에 제목 추가한 시각화
plt.figure(figsize=(12, 8))

# 관측된 데이터
plt.subplot(411)
plt.plot(decomposed.observed, label='Observed')
plt.title('Observed Sales Data')
plt.legend(loc='upper left')

# 추세 데이터
plt.subplot(412)
plt.plot(decomposed.trend, label='Trend')
plt.title('Trend in Sales Data')
plt.legend(loc='upper left')

# 계절성 데이터
plt.subplot(413)
plt.plot(decomposed.seasonal, label='Seasonal')
plt.title('Seasonality in Sales Data')
plt.legend(loc='upper left')

# 잔차 데이터
plt.subplot(414)
plt.plot(decomposed.resid, label='Residual')
plt.title('Residuals in Sales Data')
plt.legend(loc='upper left')

# 레이아웃 조정
plt.tight_layout()

# 결과 출력
plt.show()

# 최근 3개월간의 데이터를 사용하여 클러스터링을 수행
recent_3_months = sales_data.columns[-90:]  # 최근 90일 (약 3개월) 선택
sales_recent_3_months = sales_data[recent_3_months].sum(axis=1)  # 제품별 총 판매 금액 계산

# 데이터 스케일링
scaler = StandardScaler()
sales_scaled = scaler.fit_transform(sales_recent_3_months.values.reshape(-1, 1))

# K-means 클러스터링 실행
kmeans = KMeans(n_clusters=3, random_state=0)  # 클러스터 수는 임의로 3개로 설정
clusters = kmeans.fit_predict(sales_scaled)

# 클러스터링 결과 추가
sales_data['Cluster'] = clusters

# 클러스터별 판매 금액 분포 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y=sales_recent_3_months, data=sales_data)
plt.title('Sales Distribution by Cluster (Last 3 Months)')
plt.ylabel('Total Sales Amount')
plt.xlabel('Cluster')
plt.show()

# 브랜드별 총 언급량 계산
brand_keyword_total = brand_keyword_data.iloc[:, 1:].sum(axis=1)

# 브랜드별 총 판매 금액 계산
sales_data['Total Sales'] = sales_data.iloc[:, 6:].sum(axis=1)
sales_brand_total = sales_data.groupby('브랜드')['Total Sales'].sum()

# 계산된 데이터 확인
brand_keyword_total_sample = brand_keyword_total.head()
sales_brand_total_sample = sales_brand_total.head()
print(brand_keyword_total_sample, sales_brand_total_sample)

# 데이터 병합하기 위한 인덱스 재설정
brand_keyword_total_df = brand_keyword_total.reset_index(name='Total Keyword Count')
sales_brand_total_df = sales_brand_total.reset_index()

# brand_keyword_data의 브랜드 열을 인덱스로 사용한 전체 총합 계산
brand_keyword_total_indexed = brand_keyword_data.set_index('브랜드').iloc[:, 1:].sum(axis=1)

# 두 데이터프레임 병합
merged_data_corrected = pd.merge(
    brand_keyword_total_indexed.reset_index(),
    sales_brand_total.reset_index(),
    on='브랜드',
    how='inner'
)
# 열 이름 변경
merged_data_corrected.rename(columns={0: 'Total Keyword Count'}, inplace=True)

# 병합된 데이터프레임의  행 확인
print(merged_data_corrected.head())

# 상관 계수 계산
correlation = merged_data_corrected[['Total Keyword Count', 'Total Sales']].corr()

# 상관 계수 출력
print(correlation)

# 회귀 분석을 위한 데이터 준비
X = merged_data_corrected['Total Keyword Count'].values.reshape(-1, 1)
y = merged_data_corrected['Total Sales'].values

# X와 y의 내용 및 크기 확인
X_sample = X[:5]
y_sample = y[:5]
X_size = X.shape
y_size = y.shape

print(X_sample, y_sample, X_size, y_size)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 결과 출력
print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Regression Coefficients:", model.coef_)
print("Regression Intercept:", model.intercept_)
