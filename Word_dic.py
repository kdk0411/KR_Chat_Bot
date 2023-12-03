import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

csv_Path = 'ChatbotData.csv'
df = pd.read_csv(csv_Path)

qe = df['Q']
aw = df['A']

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2


class WordVocab():
  def __init__(self):
    self.word2index = {
      '<PAD>': PAD_TOKEN,
      '<SOS>': SOS_TOKEN,
      '<EOS>': EOS_TOKEN,
    }
    self.word2count = {}
    self.index2word = {
      PAD_TOKEN: '<PAD>',
      SOS_TOKEN: '<SOS>',
      EOS_TOKEN: '<EOS>'
    }

    self.n_words = 3  # PAD, SOS, EOS 포함

  def add_sentence(self, sentence):
    for word in sentence.split(' '):
      self.add_word(word)

  def add_word(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1
      
      
# 사용법
print(qe[10])
print(f'Pure: {qe[10]}')
lang = WordVocab()
lang.add_sentence(qe[10])
print('==='*10)
print('Word Dictionary')
print(lang.word2index)


# padding to sequences
max_length = 10
sentence_length = 6

sentence_tokens = np.random.randint(low=3, high=100, size=(sentence_length,))
sentence_tokens = sentence_tokens.tolist()
print(f'Generated Sentence: {sentence_tokens}')

sentence_tokens = sentence_tokens[:(max_length-1)]

token_length = len(sentence_tokens)

# 문장의 맨 끝부분에 <EOS> 토큰 추가
sentence_tokens.append(2)

for i in range(token_length, max_length-1):
    # 나머지 빈 곳에 <PAD> 토큰 추가
    sentence_tokens.append(0)

print(f'Output: {sentence_tokens}')
print(f'Total Length: {len(sentence_tokens)}')