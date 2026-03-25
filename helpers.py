import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from skimage import io, transform
import os.path as osp
import seaborn as sns

from diffae_med import templates

goeblue = '#153268'
midblue = '#0093c7'


def load_encoder(device):
   
   model_resolution = 128
   T_inv = "200"
   T_step = 100

   model_config = cxr128_autoenc_130M() 

   print(model_config.name)

   diffae_weight = ""
   diffae_latent = ""
   cls_weight = ""

   weight_dir_path = f'diffae/checkpoints/{model_config.name}'
   if not osp.exists(weight_dir_path): os.makedirs(weight_dir_path)
   model_weights =  '{weight_dir_path}/last.ckpt' '{model_download_path}'
   latent_weights =  '{weight_dir_path}/latent.pkl' '{latents_download_path}'

   classifer_config = cxr128_autoenc_130M()
   weight_dir_path = f'diffae/checkpoints/{classifer_config.name}'
   if not osp.exists(weight_dir_path): os.makedirs(weight_dir_path)
   checkpoint = '{weight_dir_path}/last.ckpt' '{cls_download_path}'

   device = device
   conf = cxr128_autoenc_130M()
   # print(conf.name)
   pretrained_encoder = LitModel(conf)
   state = torch.load(f'diffae/checkpoints/{conf.name}/last.ckpt', map_location='cpu', weights_only=False)
   pretrained_encoder.load_state_dict(state['state_dict'], strict=False)
   
   return pretrained_encoder



def sample_features(model, device, n_features, n_samples, x_data):

  mean_outputs = np.zeros((len(x_data), n_features))
  standard_variations = np.zeros((len(x_data), n_features))

  x_data = x_data.numpy()

   # sample individual features
  for i in range(n_features):

    feat_model = model.feature_nns[i]

    x_vals_compute = x_data[:, i]

    #valid_idx = (x_vals_compute >= min_fun(x_vals_compute)) & (x_vals_compute <= max_fun(x_vals_compute))

    #x_vals_compute = x_vals_compute[valid_idx]

    min_compute = min(x_vals_compute)
    max_compute = max(x_vals_compute)
    range_compute = torch.linspace(min_compute, max_compute, len(x_vals_compute))

    with torch.no_grad():

      model_input = range_compute.to(device).reshape(-1,1).float()

      output_mc = []

      for _ in range(n_samples):
        #bias = torch.normal(bias_mean, torch.exp(bias_log_scale))
        out, _ = feat_model.forward(model_input)
        out = out #+ bias
        output_mc.append(out)

      output = torch.stack(output_mc)

      mean_pred_batch = torch.mean(output, dim = 0)
      mean_pred_batch = mean_pred_batch - torch.mean(mean_pred_batch)
      std = torch.sqrt(torch.var(output, dim = 0))
      mean_outputs[:, i] = mean_pred_batch.squeeze().cpu().numpy()
      standard_variations[:, i] = std.squeeze().cpu().numpy()


  plus_error = [mean_outputs[:,i] + 2 * standard_variations[:,i] for i in range(n_features)]
  minus_error = [mean_outputs[:,i] - 2 * standard_variations[:,i] for i in range(n_features)]

  with sns.axes_style('whitegrid'):

    fig, axes = plt.subplots(n_features, 1, figsize=(7, 12))

    for i in range(n_features):

      axes[i].set_xlabel(r'$x$')
      axes[i].set_ylabel(r'$y$')
      axes[i].scatter(x_data[:, i], y_data[i,:], color = goeblue, s=2, alpha=0.7)
      axes[i].plot(range_compute, mean_outputs[:,i], "--",  color='deeppink', linewidth=3, label='Mean-Prediction')

      axes[i].plot(range_compute, plus_error[i], '-', color = midblue, linewidth=1, alpha=0.5)
      axes[i].plot(range_compute, minus_error[i], '-', color = midblue, linewidth=1, alpha=0.5)

      # Add error bands
      axes[i].fill_between(range_compute, minus_error[i], plus_error[i],
                          color= midblue, alpha=0.2, label='±2σ')
      # Or plot error lines instead:

      axes[i].set_title(f'Prediction {i+1}')
      axes[i].grid(True, alpha=0.3)
      axes[i].legend(loc='best')


    plt.tight_layout()

    img_save_path = f'images/'
    if not osp.exists(img_save_path): os.makedirs(img_save_path)

    fig.savefig(f'{img_save_path}.png')

    #plt.tight_layout()
    plt.show()


  return mean_outputs, standard_variations


