import os
import torch
from torch.utils.data.dataset import Dataset
import re
import pandas as pd
from konlpy.tag import Mecab
from Word_dic import WordVocab


csv_Path = 'ChatbotData.csv'
df = pd.read_csv(csv_Path)

qe = df['Q']
aw = df['A']


class TextDataset(Dataset):
  def __init__(self, csv_path, min_length=3, max_length=32):
    super(TextDataset, self).__init__()
    data_dir = 'data'

    # TOKEN 정의
    self.PAD_TOKEN = 0  # Padding 토큰
    self.SOS_TOKEN = 1  # SOS 토큰
    self.EOS_TOKEN = 2  # EOS 토큰

    self.tagger = Mecab()  # 형태소 분석기
    self.max_length = max_length  # 한 문장의 최대 길이 지정

    # CSV 데이터 로드
    df = pd.read_csv(os.path.join(data_dir, csv_path))

    # 한글 정규화
    korean_pattern = r'[^ ?,.!A-Za-z0-9가-힣+]'
    self.normalizer = re.compile(korean_pattern)

    # src: 질의, tgt: 답변
    src_clean = []
    tgt_clean = []

    # 단어 사전 생성
    wordvocab = WordVocab()

    for _, row in df.iterrows():
      src = row['Q']
      tgt = row['A']

      # 한글 전처리
      src = self.clean_text(src)
      tgt = self.clean_text(tgt)

      if len(src.split()) > min_length and len(tgt.split()) > min_length:
        # 최소 길이를 넘어가는 문장의 단어만 추가
        wordvocab.add_sentence(src)
        wordvocab.add_sentence(tgt)
        src_clean.append(src)
        tgt_clean.append(tgt)

    self.srcs = src_clean
    self.tgts = tgt_clean
    self.wordvocab = wordvocab

  def normalize(self, sentence):
    # 정규표현식에 따른 한글 정규화
    return self.normalizer.sub("", sentence)

  def clean_text(self, sentence):
    # 한글 정규화
    sentence = self.normalize(sentence)
    # 형태소 처리
    sentence = self.tagger.morphs(sentence)
    sentence = ' '.join(sentence)
    sentence = sentence.lower()
    return sentence

  def texts_to_sequences(self, sentence):
    # 문장 -> 시퀀스로 변환
    return [self.wordvocab.word2index[w] for w in sentence.split()]

  def pad_sequence(self, sentence_tokens):
    # 문장의 맨 끝 토큰은 제거
    sentence_tokens = sentence_tokens[:(self.max_length - 1)]
    token_length = len(sentence_tokens)

    # 문장의 맨 끝부분에 <EOS> 토큰 추가
    sentence_tokens.append(self.EOS_TOKEN)

    for i in range(token_length, (self.max_length - 1)):
      # 나머지 빈 곳에 <PAD> 토큰 추가
      sentence_tokens.append(self.PAD_TOKEN)
    return sentence_tokens

  def __getitem__(self, idx):
    inputs = self.srcs[idx]
    inputs_sequences = self.texts_to_sequences(inputs)
    inputs_padded = self.pad_sequence(inputs_sequences)

    outputs = self.tgts[idx]
    outputs_sequences = self.texts_to_sequences(outputs)
    outputs_padded = self.pad_sequence(outputs_sequences)

    return torch.tensor(inputs_padded), torch.tensor(outputs_padded)

  def __len__(self):
    return len(self.srcs)
  
  
# 사용법
# 한 문장의 최대 단어길이를 25로 설정
MAX_LENGTH = 25

dataset = TextDataset('ChatbotData.csv', min_length=3, max_length=MAX_LENGTH)

# 10번째 데이터 임의 추출
x, y = dataset[10]

print(f'x shape: {x.shape}')
print(x)

print(f'y shape: {y.shape}')
print(y)