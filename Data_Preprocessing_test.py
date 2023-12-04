import re
import pandas as pd
from konlpy.tag import Mecab, Okt
# from py_hanspell.hanspell import spell_checker
# 네이버 맞춤법 검사기를 토대로 만든 spell_checker를 가져와 사용합니다.
# 현재 오류로 인해서 사용을 못하고 있다.

csv_Path = 'ChatbotData.csv'
df = pd.read_csv(csv_Path)

qe = df['Q']
aw = df['A']

# 한글 정규화
# 한영, 숫자, 공백, ?!.,를 제외한 나머지 문자 제거
kr_pattern = r'[^ ?,.!A-Za-z0-9가-힣+]'

# 패턴 컴파일
normalizer = re.compile(kr_pattern)
# print(normalizer)
#re.compile('[^ ?,.!A-Za-z0-9가-힣+]')

# print(f'Before : {qe[5]}')
# print(f'After : {normalizer.sub("", qe[5])}')
# Before : SD카드 망가졌어
# After : SD카드 망가졌어
# print(f'Before : {aw[5]}')
# print(f'After : {normalizer.sub("", aw[5])}')
# Before : 다시 새로 사는 게 마음 편해요.
# After : 다시 새로 사는 게 마음 편해요.

def normalize(sentence):
  return normalizer.sub("", sentence)

normalize(qe[10])

# 한글 형태소 분석
# mecab은 윈도우 환경에서 자동으로 설치가 되지 않는다.
# Local C:Drive에 설치 후 Mecab을 가져올때 아래와 같이 가져오면 된다.
mecab = Mecab('C:/mecab/mecab-ko-dic')
okt = Okt()
# Mecab
# print(mecab.morphs(normalize(qe[10])))
# Out -> ['SNS', '보', '면', '나', '만', '빼', '고', '다', '행복', '해', '보여']
# Okt
# print(okt.morphs(normalize(aw[10])))
# Out -> ['자랑', '하는', '자리', '니까', '요', '.']

def clean_text(sentence, tagger):
  # 맞춤법 검사 이후 넣는다.
  # sentence_check = spell_checker.check(sentence)
  # sentence = sentence_check.ckecked
  sentence = normalize(sentence)
  sentence = tagger.morphs(sentence)
  sentence = ' '.join(sentence)
  sentence = sentence.lower()
  return sentence

# 코드 사용법
print(clean_text(qe[10], okt))
# out -> sns 보면 나 만 빼고 다 행복 해보여
print(clean_text(aw[10], okt))
# out -> 자랑 하는 자리 니까 요 .
print(len(qe), len(aw)) # -> 11823 11823
print(qe[:5])
print(aw[:5])