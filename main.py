import pandas as pd
import numpy as np

from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('./data/train.csv')
data.head()
cat_df_list = []

# Iterate over the rows of the data
for index, row in tqdm(data.iterrows()):
    cat_list = np.tile(row[:6].values, (len(row[6:]), 1))  # 첫 6개 데이터를 반복
    cat_df = pd.DataFrame(cat_list, columns=data.columns[:6])  # 반복된 데이터를 데이터프레임으로 변환
    cat_df['판매량'] = row[6:].values  # 판매량 데이터 추가
    cat_df_list.append(cat_df)

cat_df = pd.concat(cat_df_list, axis=0)
cat_df.reset_index(drop=True, inplace=True)
# print(cat_df)
