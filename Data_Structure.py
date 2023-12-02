import pandas as pd

csv_Path = 'ChatbotData.csv'
df = pd.read_csv(csv_Path)
# print(df.head())
# Out
#                  Q            A  label
# 0           12시 땡!   하루가 또 가네요.      0
# 1      1지망 학교 떨어졌어    위로해 드립니다.      0
# 2     3박4일 놀러가고 싶다  여행은 언제나 좋죠.      0
# 3  3박4일 정도 놀러가고 싶다  여행은 언제나 좋죠.      0
# 4          PPL 심하네   눈살이 찌푸려지죠.      0

qe = df['Q']
aw = df['A']
print(qe.head())
print(aw.head())