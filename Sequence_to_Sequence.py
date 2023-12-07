import random
import torch
import torch.nn as nn
from Data_preprocessing import TextDataset

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, device):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device

  def forward(self, inputs, outputs, teacher_forcing_ratio=0.5):
    # inputs : (batch_size, sequence_length)
    # outputs: (batch_size, sequence_length)

    batch_size, output_length = outputs.shape
    output_num_vocabs = self.decoder.num_vocabs

    # 리턴할 예측된 outputs를 저장할 임시 변수
    # (sequence_length, batch_size, num_vocabs)
    predicted_outputs = torch.zeros(output_length, batch_size, output_num_vocabs).to(self.device)

    # 인코더에 입력 데이터 주입, encoder_output은 버리고 hidden_state 만 살립니다.
    # 여기서 hidden_state가 디코더에 주입할 context vector 입니다.
    # (Bidirectional(1) x number of layers(1), batch_size, hidden_size)
    _, decoder_hidden = self.encoder(inputs)

    # (batch_size) shape의 SOS TOKEN으로 채워진 디코더 입력 생성
    decoder_input = torch.full((batch_size,), SOS_TOKEN, device=self.device)

    # 순회하면서 출력 단어를 생성합니다.
    # 0번째는 SOS TOKEN이 위치하므로, 1번째 인덱스부터 순회합니다.
    for t in range(0, output_length):
      # decoder_input : 디코더 입력 (batch_size) 형태의 SOS TOKEN로 채워진 입력
      # decoder_output: (batch_size, num_vocabs)
      # decoder_hidden: (Bidirectional(1) x number of layers(1), batch_size, hidden_size), context vector와 동일 shape
      decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

      # t번째 단어에 디코더의 output 저장
      predicted_outputs[t] = decoder_output

      # teacher forcing 적용 여부 확률로 결정
      # teacher forcing 이란: 정답치를 다음 RNN Cell의 입력으로 넣어주는 경우. 수렴속도가 빠를 수 있으나, 불안정할 수 있음
      teacher_force = random.random() < teacher_forcing_ratio

      # top1 단어 토큰 예측
      top1 = decoder_output.argmax(1)

      # teacher forcing 인 경우 ground truth 값을, 그렇지 않은 경우, 예측 값을 다음 input으로 지정
      decoder_input = outputs[:, t] if teacher_force else top1

    return predicted_outputs.permute(1, 0, 2)  # (batch_size, sequence_length, num_vocabs)로 변경