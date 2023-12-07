import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self, num_voca, hidden_size, embedding_dim, num_layer):
    super(Encoder, self).__init__()
    
    # 단어 개수, 레이어, GRU 정의
    self.num_voca = num_voca
    self.embedding = nn.Embedding(num_voca, embedding_dim)
    self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layer, bidirectional=False)

  def forward(self, x):
    x = self.embedding(x).permute(1, 0, 2)
    output, hidden = self.gru(x)
    return output, hidden

class Decoder(nn.Module):
  def __init__(self, num_voca, hidden_size, embedding_dim, num_layers=1, dropout=0.2):
    super(Decoder, self).__init__()
    # 단어사전 개수
    self.num_vocabs = num_voca
    self.embedding = nn.Embedding(num_voca, embedding_dim)
    self.dropout = nn.Dropout(dropout)
    self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=False)

    # 최종 출력은 단어사전의 개수
    self.fc = nn.Linear(hidden_size, num_voca)

  def forward(self, x, hidden_state):
    x = x.unsqueeze(0)  # (1, batch_size) 로 변환
    embedded = F.relu(self.embedding(x))
    embedded = self.dropout(embedded)
    output, hidden = self.gru(embedded, hidden_state)
    output = self.fc(output.squeeze(0))  # (sequence_length, batch_size, hidden_size(32) x bidirectional(1))
    return output, hidden

def train(model, data_loader, optimizer, loss_fn, device):
  model.train()
  running_loss = 0

  for x, y in data_loader:
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()

    # output: (batch_size, sequence_length, num_vocabs)
    output = model(x, y)
    output_dim = output.size(2)

    # 1번 index 부터 슬라이싱한 이유는 0번 index가 SOS TOKEN 이기 때문
    # (batch_size*sequence_length, num_vocabs) 로 변경
    output = output.reshape(-1, output_dim)

    # (batch_size*sequence_length) 로 변경
    y = y.view(-1)

    # Loss 계산
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * x.size(0)

  return running_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, device):
  model.eval()

  eval_loss = 0

  with torch.no_grad():
    for x, y in data_loader:
      x, y = x.to(device), y.to(device)
      output = model(x, y)
      output_dim = output.size(2)
      output = output.reshape(-1, output_dim)
      y = y.view(-1)

      # Loss 계산
      loss = loss_fn(output, y)

      eval_loss += loss.item() * x.size(0)

  return eval_loss / len(data_loader)

def sequence_to_sentence(sequences, index2word):
  outputs = []
  for p in sequences:

    word = index2word[p]
    if p not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
      outputs.append(word)
    if word == EOS_TOKEN:
      break
  return ' '.join(outputs)

def random_evaluation(model, dataset, index2word, device, n=10):

  n_samples = len(dataset)
  indices = list(range(n_samples))
  np.random.shuffle(indices)  # Shuffle
  sampled_indices = indices[:n]  # Sampling N indices

  # 샘플링한 데이터를 기반으로 DataLoader 생성
  sampler = SubsetRandomSampler(sampled_indices)
  sampled_dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

  model.eval()
  with torch.no_grad():
    for x, y in sampled_dataloader:
      x, y = x.to(device), y.to(device)
      output = model(x, y, teacher_forcing_ratio=0)
      # output: (number of samples, sequence_length, num_vocabs)

      preds = output.detach().cpu().numpy()
      x = x.detach().cpu().numpy()
      y = y.detach().cpu().numpy()

      for i in range(n):
        print(f'질문   : {sequence_to_sentence(x[i], index2word)}')
        print(f'답변   : {sequence_to_sentence(y[i], index2word)}')
        print(f'예측답변: {sequence_to_sentence(preds[i].argmax(1), index2word)}')
        print('===' * 10)