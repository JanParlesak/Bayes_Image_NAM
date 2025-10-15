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



class BayesFeature(nn.Module):
  def __init__(self, hid_dim=[], dropout_rate = 0.0, prior_scale= .1, activation_function = nn.LeakyReLU()):
    super(BayesFeature, self).__init__()

    self.activation = activation_function

    self.layer_sizes = [1] + hid_dim + [1]

    self.layers = nn.ModuleList([
        BayesLinear(self.layer_sizes[idx - 1], self.layer_sizes[idx], weight_prior_sigma = prior_scale) for idx in
                      range(1, len(self.layer_sizes))])

    self.dropout = torch.nn.Dropout(p = dropout_rate)

  def forward(self, x):

    kl_sum = 0

    out, kl = self.layers[0](x)
    kl_sum += kl
    out = self.activation(out)

    for layer in self.layers[1:-1]:
      out, kl = layer(out)
      kl_sum += kl
      out = self.activation(out)
      out = self.dropout(out)

    out, kl = self.layers[-1](out)
    kl_sum += kl

    return out, kl_sum
  


class BayesNAM(nn.Module):
  def __init__(self,
                n_features,   # number of neurons in first layer
                hidden_units = [],  # tuple of numbers of hidden units
                dropout_rate = 0.0,
                feature_dropout_rate = 0.0,
                activation = nn.LeakyReLU(),
                prior_scale = .1,
                return_output_lis = False
                ):
      super().__init__()


      self.samples = {'bias' : None}

      #self.shallow_units = shallow_units
      self.hidden_units = hidden_units
      self.activation = activation

      self.feature_dropout_rate = feature_dropout_rate

      self.feature_dropout = torch.nn.Dropout(p=self.feature_dropout_rate)

      self.n_features = n_features
      self.return_output_lis = return_output_lis

      self.feature_nns = nn.ModuleList([
          BayesFeature(hid_dim = hidden_units, dropout_rate= dropout_rate, prior_scale= prior_scale, activation_function = activation)
          for i in range(n_features)
      ])

      self.bias = nn.Parameter(torch.zeros(1))   #does the bias need a prior? probably yes

      self.bias_prior_mu = 0.
      self.bias_prior_sigma = .1
      self.bias_mean = Parameter(torch.rand(1) -0.5) # intialize bias mean if given
      self.lbias_sigma = Parameter(torch.log(prior_scale* torch.ones(1)))


  def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        kl = sigma_p - sigma_q + (torch.exp(sigma_q)**2 + (mu_q - mu_p)**2) / (2 * math.exp(sigma_p)**2) - 0.5  # Kullback Leibler divergence for two normals

        return kl.mean()

  def getSampledBias(self):
        return self.samples['bias']


  def forward(self, f):

    #eta = self.bias
    output_lis = []
    kl_total = 0

    for feature, mod in zip(f.T, self.feature_nns):

      feature = feature.unsqueeze(-1)
      ri, kl = mod(feature)
      kl_total += kl
      output_lis.append(ri)

    #self.samples['bias'] = self.bias_mean + torch.exp(self.lbias_sigma) * torch.randn_like(self.lbias_sigma)
    #kl_total += self.kl_div(self.bias_mean, self.lbias_sigma, self.bias_prior_mu, self.bias_prior_sigma)


    conc_out = torch.cat(output_lis, dim=-1)
    dropout_out = self.feature_dropout(conc_out)
    out = torch.sum(dropout_out, dim=-1) + self.bias #+ self.samples['bias']


    return out, kl_total
    

class BayesSkipBlock(nn.module):
   def __init__(self, in_features, out_features, prior_scale = 0.1):
    super(BayesSkipBlock, self).__init__()


    self.linear_in = BayesLinear(in_features = in_features, out_features = 100, weight_prior_sigma = prior_scale)
    self.linear_out = BayesLinear(in_features = 100, out_features = out_features, weight_prior_sigma = prior_scale)

    self.act = torch.nn.LeakyReLU()

    self.bn1 = nn.BatchNorm1d(100, affine = True)
    self.bn2 = nn.BatchNorm1d(out_features, affine = True)


   def forward(self, x):

    kl_sum = 0
    x0 = x

    x, kl = self.linear_in(x)
    kl_sum += kl
    x = self.bn1(x)
    x = self.act(x)

    x, kl = self.linear_out(x)
    kl_sum += kl
    x = self.bn2(x)

    x = x0 + x

    x = self.act(x)

    return x, kl_sum
   



class BayesResFeature(nn.Module):
  def __init__(self, hid_dim=[50,100,100,50], dropout_rate = 0.0, prior_scale= .1, activation_function = nn.LeakyReLU()):
    super(BayesResFeature, self).__init__()

    self.activation = activation_function

    self.input_layer = BayesLinear(1, hid_dim[0])

    self.layers = nn.ModuleList([
        BayesSkipBlock(hid_dim[idx], hid_dim[idx], weight_prior_sigma = prior_scale) for idx in
                      range(1, len(hid_dim))])
    
    self.output_layer = BayesLinear(hid_dim[-1], 1)

    self.dropout = torch.nn.Dropout(p = dropout_rate)

    self.bn1 = nn.BatchNorm1d(hid_dim[0], affine = True)

  def forward(self, x):

    kl_sum = 0

    out, kl = self.input_layer(x)
    kl_sum += kl
    out = self.bn1(out)
    out = self.activation(out)
    out = self.dropout(out)

    for layer in self.layers:
      out, kl = layer(out)
      kl_sum += kl

    out, kl = self.output_layer(out)
    kl_sum += kl

    return out, kl_sum
  




class BayesRes(nn.Module):
  def __init__(self,
                n_features,   # number of neurons in first layer
                hidden_units = [50,100,100,50],  # tuple of numbers of hidden units
                dropout_rate = 0.0,
                feature_dropout_rate = 0.0,
                activation = nn.LeakyReLU(),
                prior_scale = .1,
                return_output_lis = False
                ):
      super().__init__()


      self.samples = {'bias' : None}

      #self.shallow_units = shallow_units
      self.hidden_units = hidden_units
      self.activation = activation

      self.feature_dropout_rate = feature_dropout_rate

      self.feature_dropout = torch.nn.Dropout(p=self.feature_dropout_rate)

      self.n_features = n_features
      self.return_output_lis = return_output_lis

      self.feature_nns = nn.ModuleList([
          BayesResFeature(hid_dim = hidden_units, dropout_rate= dropout_rate, prior_scale= prior_scale, activation_function = activation)
          for i in range(n_features)
      ])

      self.bias = nn.Parameter(torch.zeros(1))   #does the bias need a prior? probably yes

      self.bias_prior_mu = 0.
      self.bias_prior_sigma = .1
      self.bias_mean = Parameter(torch.rand(1) -0.5) # intialize bias mean if given
      self.lbias_sigma = Parameter(torch.log(prior_scale* torch.ones(1)))


  def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        kl = sigma_p - sigma_q + (torch.exp(sigma_q)**2 + (mu_q - mu_p)**2) / (2 * math.exp(sigma_p)**2) - 0.5  # Kullback Leibler divergence for two normals

        return kl.mean()

  def getSampledBias(self):
        return self.samples['bias']


  def forward(self, f):

    #eta = self.bias
    output_lis = []
    kl_total = 0

    for feature, mod in zip(f.T, self.feature_nns):

      feature = feature.unsqueeze(-1)
      ri, kl = mod(feature)
      kl_total += kl
      output_lis.append(ri)

    #self.samples['bias'] = self.bias_mean + torch.exp(self.lbias_sigma) * torch.randn_like(self.lbias_sigma)
    #kl_total += self.kl_div(self.bias_mean, self.lbias_sigma, self.bias_prior_mu, self.bias_prior_sigma)


    conc_out = torch.cat(output_lis, dim=-1)
    dropout_out = self.feature_dropout(conc_out)
    out = torch.sum(dropout_out, dim=-1) + self.bias #+ self.samples['bias']


    return out, kl_total