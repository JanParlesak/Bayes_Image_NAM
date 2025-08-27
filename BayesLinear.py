import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class BayesLinear(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias = True,
                 weight_prior_mu = 0,
                 weight_prior_sigma = 0.1,
                 bias_prior_mu = 0,
                 bias_prior_sigma = 1.):

        super(BayesLinear, self).__init__()

        self.samples = {'weights' : None, 'bias' : None}

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_prior_mu = weight_prior_mu
        self.weight_prior_sigma = weight_prior_sigma
        self.l_weight_prior_sigma = torch.log(torch.tensor(weight_prior_sigma))

        self.weights_mu = Parameter(torch.rand(out_features, in_features)-0.5)
        self.lweights_sigma = Parameter(torch.log(weight_prior_sigma*torch.ones(out_features, in_features)))

        if self.bias:
            self.bias_prior_mu = bias_prior_mu
            self.bias_prior_sigma = bias_prior_sigma
            self.bias_mean = Parameter(torch.rand(out_features)-0.5)
            self.lbias_sigma = Parameter(torch.log(bias_prior_sigma* torch.ones(out_features)))



    def getSampledWeights(self):
        return self.samples['weights']

    def getSampledBias(self):
        return self.samples['bias']

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        kl = sigma_p - sigma_q + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *(sigma_p**2)) - 0.5

        return kl.mean()


    def forward(self, x):

        self.samples['weights'] = self.weights_mu + torch.exp(self.lweights_sigma) * torch.randn_like(self.lweights_sigma)

        kl = self.kl_div(self.weights_mu, self.lweights_sigma, self.weight_prior_mu, self.l_weight_prior_sigma)

        if self.bias:
            kl += self.kl_div(self.bias_mean, self.lbias_sigma, self.bias_prior_mu, self.bias_prior_sigma)

        if self.bias:
            self.samples['bias'] = self.bias_mean + torch.exp(self.lbias_sigma) * torch.randn_like(self.lbias_sigma)

        out = F.linear(x, self.samples['weights'], self.samples['bias'] if self.bias else None)


        return out, kl