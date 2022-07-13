# -*- coding: utf-8 -*-
"""CoBERL-model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/144tOCZVfDs31GTa72p-_BGappRR63LH9
"""

import torch
from torch import nn
from torchvision import models

import torch
import torch.nn as  nn
import torch.nn.functional as F

# !rm -r Transformer-RL/
# !git clone -b dev https://github.com/RodkinIvan/Transformer-RL

# Commented out IPython magic to ensure Python compatibility.
# %cd Transformer-RL/

# !ls

from model.encoder import ResNet47
from GTrXL.gtrxl import GTrXL

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.resnet47 = ResNet47(512)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(in_features=512, out_features=448)

  def forward(self, img, encoded):
    x = self.resnet47(img)
    x = self.relu(x)
    x = self.fc1(x)
    x = self.relu(x)


    y = nn.Linear(in_features=encoded.shape[1], out_features=64)(encoded)
    y = self.relu(y)
    print(y.shape, x.shape)
    return torch.concat((y, x), dim=1).reshape((1, 1, 512))


class CoBERL(nn.Module):
  def __init__(self):
    super(CoBERL, self).__init__()
    self.encoder = Encoder()

    self.gtrxl = GTrXL(input_dim=512,
            head_dim=64,
            embedding_dim=512,
            head_num=8,
            mlp_num=2,
            layer_num=8,
            memory_len=64,
            activation=nn.GELU()
    )

    self.gru = nn.GRU(input_size=512, hidden_size=512, num_layers=1)
    self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1)


  def forward(self, input, prev_reward_action=None):
    # The previous reward and one-hot encoded action are concatenated and projected
    # with a linear layer into a 64-dimensional vector
    if prev_reward_action is None:
      prev_reward_action = torch.rand(size=(1, 64))

    y = self.encoder(input, prev_reward_action)
    print("fully encoded", y.shape)
    x = self.gtrxl(y)['logit'].reshape((1, 512))
    y = y.reshape((1, 512))
    print("x y ", x.shape, y.shape)
    z, h_n = self.gru(y, x)
    print("GRU", z.shape)
    out, (h_n, c_n) = self.lstm(z)
    print('lstm', out.shape, y.shape)
    v_input = torch.concat((out, y), dim=1).flatten()
    print('concat', v_input.shape)
    return v_input



if __name__ == "__main__":
  model = CoBERL()
  input = torch.rand(size=(1, 3, 240, 240))
  output = model(input)
  output