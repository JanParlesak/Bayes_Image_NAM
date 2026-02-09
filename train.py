from modules import *
from helpers import *

import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.optim import Adam
import sklearn
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score, # Import the rest of the metrics
)

import time
import tqdm 
from tqdm import tqdm



goeblue = '#153268'
midblue = '#0093c7'


def validate(model, val_loader, loss_fun, kl_weight, n_samples, batch_size):

    val_loss = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):

            input, target = batch

            target = target.to(device)
            input = input.to(device)

            output = []
            kl_div = []

            for sample in range(n_samples):
                out, kl = model(input)
                output.append(out)
                kl_div.append(kl)

            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)
            scaled_kl = kl_loss * kl_weight / batch_size
            log_lik_loss = loss_fun(mean_pred.squeeze(-1), target)
            loss = log_lik_loss + scaled_kl
            val_loss.append(loss.cpu())

        mean_loss = np.mean(np.array(val_loss))

    return mean_loss


def train(model, optimizer, loss_fun, trainset, valset, device, n_epochs, n_samples, kl_weight = 0.1, batch_size = 1, early_stopping = True, n_epochs_early_stopping = 50, save_path = None, print_mod = 1):

    loss_lis = []
    overall_loss = []
    val_loss = []

    if early_stopping == True:
      n_early_stopping = n_epochs_early_stopping
      past_val_losses = []

    model = model.to(device)

    for epoch in range(n_epochs):

        for i, batch in enumerate(trainset):

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            output = []
            kl_div = []

            for _ in range(n_samples):
                out, kl = model(x)
                output.append(out)
                kl_div.append(kl)


            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)

            #print(kl_loss)

            mean_pred = mean_pred.squeeze(-1)


            loss = loss_fun(mean_pred, y)
            scaled_kl = kl_loss * kl_weight / batch_size
            loss += scaled_kl  #ELBO Loss add if loos_fun is negative log_likelihood

            #loss = loss_fun(out, y)
            #loss += kl * 0.1     # Why does this improve training so much? kl / batch_size and add sampling again


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_lis.append(loss.cpu().detach())

        if epoch % print_mod == 0:

            mean_loss = np.mean(np.array(loss_lis))
            overall_loss += mean_loss
            loss_lis = []

            validation_loss = validate(model = model, val_loader = valset, loss_fun = loss_fun, kl_weight = kl_weight, n_samples = n_samples, batch_size= batch_size)

            val_loss.append(validation_loss)


            print(f'Epoch nr {epoch}: mean_train_loss = {mean_loss}, mean_valid_loss = {validation_loss}')



            if early_stopping:

              if len(past_val_losses) == 0 or validation_loss < min(past_val_losses):
                print("save model")
                torch.save(model.state_dict(), save_path)

              if len(past_val_losses) >= n_early_stopping:
                if validation_loss > max(past_val_losses):
                  print(f"Early stopping because the median validation loss has not decreased since the last {n_early_stopping} epochs")
                  return overall_loss
                else:
                  past_val_losses = past_val_losses[1:] + [validation_loss]
              else:
                past_val_losses = past_val_losses + [validation_loss]

    print(kl_loss)

    return overall_loss



def sample_features(model, n_features, n_samples, x_data, y_data, function_list):

  mean_outputs = np.zeros((len(x_data), n_features))
  standard_variations = np.zeros((len(x_data), n_features))

  #bias_mean = model.bias_mean
  #bias_log_scale = model.lbias_sigma
  #bias = model.bias.item()

  max_fun = lambda x: np.quantile(x, q = 1)
  min_fun = lambda x: np.quantile(x, q = 0)


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

    #plt.tight_layout()
    plt.show()


  return mean_outputs, standard_variations



def validate_images(model, device, mode, val_loader, loss_fun, kl_weight, batch_size, n_samples):

    val_loss = []

    target_lis = []
    pred_lis = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):

            images, features, target = batch

            images = images.to(device)
            target = target.to(device)
            features = features.to(device)

            output = []
            kl_div = []

            for sample in range(n_samples):
                out, kl = model(images, features)
                output.append(out)
                kl_div.append(kl)

            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)
            scaled_kl = kl_loss * kl_weight / batch_size
            log_lik_loss = loss_fun(mean_pred.squeeze(-1), target)
            loss = log_lik_loss + scaled_kl
            val_loss.append(loss.cpu())

            target_lis.append(y.detach().cpu())
            pred_lis.append(torch.round(torch.sigmoid(mean_pred)).detach().cpu() if mode == "classification" else 
                            mean_pred.detach().cpu())

        mean_loss = np.mean(np.array(val_loss))

        target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)

        if mode == "classification":
          acc = accuracy_score(target_ten, pred_ten)
          recall = recall_score(target_ten, pred_ten)
          precision = precision_score(target_ten, pred_ten)
          f1 = f1_score(target_ten, pred_ten)

          return mean_loss, acc, recall , precision, f1
        else:
           var_exp = var_exp_score(pred_ten, target_ten)
           mad_exp = mad_explained(pred_ten, target_ten)
           r_score = coef_det(pred_ten, target_ten)

           return mean_loss, var_exp, mad_exp, r_score


    


def train_images(model, optimizer, loss_fun, trainset, valset, device, n_epochs, n_samples, mode, batch_size = 256, kl_weight = 0.1, early_stopping = True, n_epochs_early_stopping = 50, save_path = None, print_mod = 1):

    loss_lis = []
    overall_loss = []
    val_loss = []
    target_lis = []
    pred_lis = []


    if early_stopping == True:
      n_early_stopping = n_epochs_early_stopping
      past_val_losses = []

    model = model.to(device)
    model.train()

    for epoch in range(n_epochs):
        start = time.time()
        for i, batch in enumerate(tqdm(trainset)):

            images, features, target = batch

            images = images.to(device)
            target = target.to(device)
            features = features.to(device)

            output = []
            kl_div = []

            for _ in range(n_samples):
                out, kl = model(images, features)
                output.append(out)
                kl_div.append(kl)


            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)

            #print(kl_loss)

            mean_pred = mean_pred.squeeze(-1)


            loss = loss_fun(mean_pred, target)
            scaled_kl = kl_loss * kl_weight / batch_size
            loss += scaled_kl  #ELBO Loss add if loos_fun is negative log_likelihood

            #loss = loss_fun(out, y)
            #loss += kl * 0.1     # Why does this improve training so much? kl / batch_size and add sampling again


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_lis.append(loss.cpu().detach())
            target_lis.append(target.detach().cpu())
            pred_lis.append(torch.round(torch.sigmoid(mean_pred)).detach().cpu() if mode == "classification" else 
                            mean_pred.detach().cpu())

        if epoch % print_mod == 0:
            
            end = time.time()
            time_delta = end - start

            mean_loss = np.mean(np.array(loss_lis))
            overall_loss += mean_loss
            loss_lis = []

            target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)

            if mode == "classification":
              acc = accuracy_score(target_ten, pred_ten)
              recall = recall_score(target_ten, pred_ten)
              precision = precision_score(target_ten, pred_ten)
              f1 = f1_score(target_ten, pred_ten)

              mean_loss_val, acc_val, recall_val, precision_val, f1_val  = validate_images(model = model, device = device, mode = mode, 
                                              val_loader = valset, loss_fun = loss_fun, kl_weight = kl_weight, batch_size = batch_size, n_samples = n_samples)
              val_loss.append(mean_loss_val)

              print(f'Epoch nr {epoch}: mean_train_loss = {mean_loss}, train_acc = {acc},train_recall = {recall}, train_precision = {precision}, train_f1 = {f1}, elapsed time: {time_delta}')
              print(f'Epoch nr {epoch}: mean_valid_loss = {mean_loss_val}, val_accuracy = {acc_val}, val_recall = {recall_val}, val_precision = {precision_val},  val_f1 = {f1_val}')


            else:

              var_exp = var_exp_score(pred_ten, target_ten)
              mad_exp = mad_explained(pred_ten, target_ten)
              r_score = coef_det(pred_ten, target_ten)

              mean_loss_val, var_exp_val, mad_exp_val, r_score_val = validate_images(model = model, device = device, mode = mode, 
                                                val_loader = valset, loss_fun = loss_fun, kl_weight = kl_weight, batch_size = batch_size, n_samples = n_samples)

              val_loss.append(mean_loss_val)

              print(f'Epoch nr {epoch}: mean_train_loss = {mean_loss}, train_acc = {var_exp}, train_recall = {mad_exp}, train_precision = {r_score}, elapsed time: {time_delta}')
              print(f'Epoch nr {epoch}: mean_valid_loss = {mean_loss_val}, val_accuracy = {var_exp_val}, val_recall = {mad_exp_val}, val_precision = {r_score_val}')



            if early_stopping:

              if len(past_val_losses) == 0 or mean_loss_val < min(past_val_losses):
                print("save model")
                torch.save(model.state_dict(), save_path)

              if len(past_val_losses) >= n_early_stopping:
                if mean_loss_val > max(past_val_losses):
                  print(f"Early stopping because the median validation loss has not decreased since the last {n_early_stopping} epochs")
                  return overall_loss
                else:
                  past_val_losses = past_val_losses[1:] + [mean_loss_val]
              else:
                past_val_losses = past_val_losses + [mean_loss_val]

    return overall_loss



def validate_classifier(model, dataloader, loss_fun):
    val_loss_lis = []

    target_lis = []
    pred_lis = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

          x, f, y = batch
          x = x.to(device)
          f = f.to(device)
          y = y.to(device)
          pred = model(x, f)
          pred = pred.squeeze(-1)
          loss = loss_fun(pred, y)
          val_loss_lis.append(loss.cpu().detach())

          target_lis.append(y.detach().cpu())
          pred_lis.append(torch.round(torch.sigmoid(pred)).detach().cpu())

    mean_loss = np.mean(np.array(val_loss_lis))
    median_loss = np.median(np.array(val_loss_lis))

    target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)
    acc = accuracy_score(target_ten, pred_ten)
    recall = recall_score(target_ten, pred_ten)
    precision = precision_score(target_ten, pred_ten)
    f1 = f1_score(target_ten, pred_ten)
    return mean_loss, median_loss, acc, recall , precision, f1


def train_classifier(model, optimizer, loss_fun, trainset, valset, print_mod, device, n_epochs, save_path = None, early_stopping = True, n_epochs_early_stopping = 5):
    """
    train the model
    Args:
        model: The model to train
        optimizer: The used optimizer
        loss_fun: The used loss function
        trainset: The dataset to train on
        valset: The dataset to use for validation
        print_mod: Number of epochs to print result after
        device: Either "cpu" or "cuda"
        n_epochs: Number of epochs to train
        save_path: Path to save the model's state dict
        config: config file from the model to train
        sparse_ten (bool): if a sparse tensor is used for each batch
    """
    if early_stopping == True:
      n_early_stopping = n_epochs_early_stopping
      past_val_losses = []

    loss_lis = []
    target_lis = []
    pred_lis = []

    loss_lis_all = []
    val_loss_lis_all = []

    model = model.to(device)

    model.train()
    for epoch in range(n_epochs):
      start = time.time()
      for iter, batch in enumerate(tqdm(trainset)):

        x, f, y = batch
        x = x.to(device)
        f = f.to(device)
        y = y.to(device)
        pred = model(x, f)
        pred = pred.squeeze(-1)


        loss = loss_fun(pred, y)
        #print(loss)

        optimizer.zero_grad()       # clear previous gradients
        loss.backward()             # backprop

        optimizer.step()

        loss_lis.append(loss.cpu().detach())
        target_lis.append(y.detach().cpu())
        pred_lis.append(torch.round(torch.sigmoid(pred)).detach().cpu())

      if epoch % print_mod == 0:

        end = time.time()
        time_delta = end - start

        mean_loss = np.mean(np.array(loss_lis))
        median_loss = np.median(np.array(loss_lis))

        target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)
        #var_exp = var_exp_score(pred_ten, target_ten)
        #mad_exp = mad_explained(pred_ten, target_ten)
        #r_score = coef_det(pred_ten, target_ten)

        acc = accuracy_score(target_ten, pred_ten)
        recall = recall_score(target_ten, pred_ten)
        precision = precision_score(target_ten, pred_ten)
        f1 = f1_score(target_ten, pred_ten)

        target_lis = []
        pred_lis = []



        loss_lis_all += loss_lis

        loss_lis = []



        mean_loss_val, median_loss_val, acc_val, recall_val, precision_val, f1_val = validate(model, val_loader, loss_fun = loss_fun)

        val_loss_lis_all.append(mean_loss_val)



        print(f'Epoch nr {epoch}: mean_train_loss = {mean_loss}, median_train_loss = {median_loss}, train_acc = {acc}, train_recall = {recall}, train_precision = {precision}, train_f1 = {f1}, elapsed time: {time_delta}')
        print(f'Epoch nr {epoch}: mean_valid_loss = {mean_loss_val}, median_valid_loss = {median_loss_val}, val_accuracy = {acc_val}, val_recall = {recall_val}, val_precision = {precision_val},  val_f1 = {f1_val}')



        # early stopping based on median validation loss:
        if early_stopping:
          if len(past_val_losses) == 0 or mean_loss_val < min(past_val_losses):
            print("save model")
            torch.save(model.state_dict(), save_path)

          if len(past_val_losses) >= n_early_stopping:
            if mean_loss_val > max(past_val_losses):
              print(f"Early stopping because the median validation loss has not decreased since the last {n_early_stopping} epochs")
              return loss_lis_all, val_loss_lis_all
            else:
              past_val_losses = past_val_losses[1:] + [mean_loss_val]
          else:
            past_val_losses = past_val_losses + [mean_loss_val]



    return loss_lis_all, val_loss_lis_all



def train_bnaim(encoder, mode, n_features, hidden_units, dropout_rate, feature_dropout_rate, prior_scale, learning_rate, device, n_epochs, n_samples, n_post_samples):
   #load encoder
   # load data

   bayes_mlp = BayesResFeature(n_input = 512) # needs to be able to adjust import at some point
   bayes_nam = BayesNAM(n_features = n_features, hidden_units = hidden_units, dropout_rate = dropout_rate, feature_dropout_rate = feature_dropout_rate,
                        prior_scale = prior_scale)

   model = BayesImageNAM(pretrained_encoder = encoder, bayes_mlp = bayes_mlp, bayes_nam = bayes_nam)

   if mode == "classification":
      loss_function = F.binary_cross_entropy_with_logits
   else: 
      loss_function = nn.MSELoss()
    
   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

   model_save_name = 'one_model'
   path = './one_checkpoint.pt'

   loss = train_images(model = model, optimizer = optimizer, loss_fun = loss_function, trainset = train_loader_img, valset= val_loader_img, 
                           device = device, n_epochs = n_epochs, n_samples= n_samples, kl_weight= 0.01, early_stopping = True, n_epochs_early_stopping = 50, save_path = path, print_mod = 10)
   
   torch.save(model.state_dict(), path)

   sample_bnaim(model = model, n_features=n_features, n_samples = n_post_samples)


def sample_bnaim(model, path, n_features, n_samples = 100, DATA_FUNCTIONS = None):
   
   model.load_state_dict(torch.load(path))
   x_data_test, y_data_test = create_test_data
   means, variances = sample_features(model = model.bayes_feat_nam, n_features = n_features, n_samples = n_samples, x_data = x_data_test, y_data=y_data_test, function_list=DATA_FUNCTIONS) # save plots in repo

   return means, variances



   





   






