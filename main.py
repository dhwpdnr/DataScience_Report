import pandas as pd
from konlpy.tag import Okt

path = "./data/"
name = "kakao_talk.csv"

read_fileName = path + name

df = pd.read_csv(read_fileName)

deleteS = ['ㅜ', 'ㅠ', 'ㅋ', 'ㅇ', 'ㅎ', 'ㅗ']

for item in df.itertuples():
    date = item[1]
    name = item[2]
    text = item[3]
    word_list = text.split(" ")
