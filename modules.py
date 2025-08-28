import torch 
import torch.nn as nn
from . import BayesLinear


class Model(nn.Module):
  def __init__(self, in_features, out_features):
    super(Model, self).__init__()

    self.layer1 = BayesLinear(in_features = in_features, out_features=100)
    self.activation = nn.ReLU()
    self.layer2 = BayesLinear(in_features=100, out_features= out_features)

  def forward(self, x):

    kl_sum = 0

    out, kl = self.layer1(x)
    kl_sum += kl
    out = self.activation(out)

    out, kl = self.layer2(out)

    kl_sum += kl

    return out, kl_sum