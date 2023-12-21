import pandas as pd
from konlpy.tag import Okt
import operator

path = "./data/"
name = "kakao_talk.csv"

read_fileName = path + name

df = pd.read_csv(read_fileName)

deleteS = ['ㅜ', 'ㅠ', 'ㅋ', 'ㅇ', 'ㅎ', 'ㅗ']

okt = Okt()
words_freq = {}
name_freq = {}
df["Date"] = pd.to_datetime(df["Date"])
date_counts = df.groupby(df['Date'].dt.date).size()
date_counts_sorted = date_counts.sort_values(ascending=False)

print(date_counts_sorted)
# df["month"] = df["date"].dt.month
# df["day"] = df["date"].dt.day
# print(df.groupby(['year', 'month'])['count'].sum())
for item in df.itertuples():
    date = item[1]
    name = item[2]
    text = item[3]
    for d in deleteS:
        text = text.replace(d, '')
    # word_list = text.split(" ")
    # print(word_list)

    if name_freq.get(name):
        name_freq[name] += 1
    else:
        name_freq[name] = 1

    n_list = okt.nouns(text)
    for n in n_list:
        if len(n) > 1:
            if n not in list(words_freq.keys()):
                words_freq[n] = 1
            else:
                words_freq[n] = words_freq[n] + 1

sdict = sorted(words_freq.items(), key=operator.itemgetter(1), reverse=True)
ndict = sorted(name_freq.items(), key=operator.itemgetter(1), reverse=True)
# for s in sdict:
#     print(s)

# for n in ndict:
#     print(n)
