from modules import *
from helpers import *
from diffae import *
from metrics import * 
from helpers_data import *


from ray import tune
from ray.tune.schedulers import ASHAScheduler

import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.optim import Adam
import time
import tqdm 
from tqdm import tqdm
import os.path as osp
import tempfile
import json


# train NAM 

def validate_nam(model, device, mode, val_loader, loss_fun, batch_size):

    val_loss = []

    target_lis = []
    pred_lis = []
    norm_pred = []

    model.eval()

    with torch.no_grad():
        for batch in val_loader:

            features, target = batch

            target = target.to(device)
            features = features.to(device)

            pred = model(features)

            pred = pred.squeeze(-1)

            loss = loss_fun(pred, target)

            val_loss.append(loss.cpu().detach())

            target_lis.append(target.detach().cpu())
            norm_pred.append(torch.sigmoid(pred).detach().cpu())


            pred_lis.append(torch.round(torch.sigmoid(pred)).detach().cpu() if mode == "classification" else
                            pred.detach().cpu())

        mean_loss = np.mean(np.array(val_loss))

        target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)
        norm_pred_lis = torch.cat(norm_pred)


        if mode == "classification":
          pre, re, thresholds = precision_recall_curve(target_ten, norm_pred_lis) 
          auc_precision_recall = auc(re, pre)

          pre_2, re_2, thresholds_2 = roc_curve(target_ten, norm_pred_lis)
          auc_val = auc(pre_2, re_2)

          balanced_acc = balanced_accuracy_score(target_ten, pred_ten)
          recall = recall_score(target_ten, pred_ten)
          precision = precision_score(target_ten, pred_ten)


          return mean_loss, auc_precision_recall, auc_val, balanced_acc, recall, precision
          
        else:
           var_exp = var_exp_score(pred_ten, target_ten)
           mad_exp = mad_explained(pred_ten, target_ten)
           r_score = coef_det(pred_ten, target_ten)

           return mean_loss, var_exp, mad_exp, r_score



def train_nam(config):
    
    root_dir = 'data/cxr_images_ny'
    data_dir = 'data/patient_data.csv'
    
    device = config["device"]
    mode = config["mode"]
    target = config["target"]

    trainset, valset, _ , features = dataloaders(target_column = target, train_frac = 0.7, val_frac = 0.2, batch_size = config["batch_size"],
                                                 root_dir=root_dir, data_dir=data_dir)

    model = make_model(config, features.shape[-1])

    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])

    if mode == "classification":
      loss_fun = F.binary_cross_entropy_with_logits

    elif mode == "regression": 
      loss_fun = nn.MSELoss()

    overall_loss = []
    val_loss = []
    loss_lis = []

    model = model.to(device)
    model.train()

    for epoch in range(config["n_epochs"]):
        
        loss_lis = []
        target_lis = []
        pred_lis = []

        for i, batch in enumerate(tqdm(trainset)):

            features, target = batch

            target = target.to(device)
            features = features.to(device)

            pred = model(features)

            pred = pred.squeeze(-1)

            loss = loss_fun(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_lis.append(loss.cpu().detach())
            target_lis.append(target.detach().cpu())
            pred_lis.append(torch.round(torch.sigmoid(pred)).detach().cpu() if mode == "classification" else
                            pred.detach().cpu())

        

        mean_loss = np.mean(np.array(loss_lis))
        overall_loss += mean_loss
        loss_lis = []

        target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)

        if mode == "classification":

          mean_loss_val, pr_auc_val, auc_val, balanced_acc_val, recall_val, precision_val  = validate_nam(model = model, device = device, mode = mode,
                                          val_loader = valset, loss_fun = loss_fun, batch_size = config["batch_size"])
          val_loss.append(mean_loss_val)

          metrics = {"mean_valid_loss" : mean_loss_val, "AUC_PR" : pr_auc_val, "AUC" : auc_val, "Balanced Accuracy" : balanced_acc_val, "val_recall" : recall_val, "val_precision" : precision_val}


        elif mode == "regression":

          mean_loss_val, var_exp_val, mad_exp_val, r_score_val = validate_nam(model = model, device = device, mode = mode,
                                            val_loader = valset, loss_fun = loss_fun, batch_size = config["batch_size"])

          val_loss.append(mean_loss_val)

          metrics =  {"mean_valid_loss" : mean_loss_val, "val_accuracy" : var_exp_val, "val_recall" : mad_exp_val, "val_precision" : r_score_val}

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
            tune.report(metrics, checkpoint=checkpoint)

    print("Finished Training!")




# train BNAM 

def validate_bnam(model, device, mode, val_loader, loss_fun, kl_weight, batch_size, n_samples):

    val_loss = []

    target_lis = []
    pred_lis = []
    norm_pred = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):

            features, target = batch

            target = target.to(device)
            features = features.to(device)

            output = []
            kl_div = []


            for sample in range(n_samples):
                out, kl = model(features)
                output.append(out)
                kl_div.append(kl)

            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)

            mean_pred = mean_pred.squeeze(-1)
            log_lik_loss = loss_fun(mean_pred, target)

            scaled_kl = kl_loss * kl_weight / batch_size

            loss = log_lik_loss + scaled_kl
            val_loss.append(loss.cpu())

            target_lis.append(target.detach().cpu())
            norm_pred.append(torch.sigmoid(mean_pred).detach().cpu())


            pred_lis.append(torch.round(torch.sigmoid(mean_pred)).detach().cpu() if mode == "classification" else
                            mean_pred.detach().cpu())

        mean_loss = np.mean(np.array(val_loss))

        target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)
        norm_pred_lis = torch.cat(norm_pred)


        if mode == "classification":
          pre, re, thresholds = precision_recall_curve(target_ten, norm_pred_lis) #other paper uses roc_curve here
          auc_precision_recall = auc(re, pre)

          pre_2, re_2, thresholds_2 = roc_curve(target_ten, norm_pred_lis)
          auc_val = auc(pre_2, re_2)

          balanced_acc = balanced_accuracy_score(target_ten, pred_ten)
          recall = recall_score(target_ten, pred_ten)
          precision = precision_score(target_ten, pred_ten)

          return mean_loss, auc_precision_recall, auc_val, balanced_acc, recall, precision

        else:
           var_exp = var_exp_score(pred_ten, target_ten)
           mad_exp = mad_explained(pred_ten, target_ten)
           r_score = coef_det(pred_ten, target_ten)

           return mean_loss, var_exp, mad_exp, r_score





def train_bnam(config):
    
    root_dir = 'data/cxr_images_ny'
    data_dir = 'data/patient_data.csv'
    
    device = config["device"]
    mode = config["mode"]
    target = config["target"]
  
    trainset, valset, _ , features = dataloaders(target_column = target, train_frac = 0.7, val_frac = 0.2, batch_size = config["batch_size"],
                                                 root_dir=root_dir, data_dir=data_dir)

    model = make_model(config, features.shape[-1])

    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])

    if mode == "classification":
      loss_fun = F.binary_cross_entropy_with_logits

    elif mode == "regression": 
      loss_fun = nn.MSELoss()

    loss_lis = []
    overall_loss = []
    val_loss = []

    model = model.to(device)
    model.train()


    overall_loss = []
    val_loss = []
    loss_lis = []

    model = model.to(device)
    model.train()

    for epoch in range(config["n_epochs"]):
        
        start = time.time()

        loss_lis = []
        target_lis = []
        pred_lis = []

        for i, batch in enumerate(tqdm(trainset)):

            features, target = batch

            target = target.to(device)
            features = features.to(device)

            output = []
            kl_div = []

            for _ in range(config["n_samples"]):
                out, kl = model(features)
                output.append(out)
                kl_div.append(kl)


            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)

            mean_pred = mean_pred.squeeze(-1)

            loss = loss_fun(mean_pred, target)
            scaled_kl = kl_loss * config["kl_weight"] / config["batch_size"]
            loss += scaled_kl  #ELBO Loss add if loos_fun is negative log_likelihood

      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_lis.append(loss.cpu().detach())
            target_lis.append(target.detach().cpu())
            pred_lis.append(torch.round(torch.sigmoid(mean_pred)).detach().cpu() if mode == "classification" else
                            mean_pred.detach().cpu())

        mean_loss = np.mean(np.array(loss_lis))
        overall_loss += mean_loss
        loss_lis = []

        target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)

        if mode == "classification":
            
          mean_loss_val, pr_auc_val, auc_val, balanced_acc_val, recall_val, precision_val  = validate_bnam(model = model, device = device, mode = mode,
                                          val_loader = valset, loss_fun = loss_fun, kl_weight = config["kl_weight"], batch_size = config["batch_size"], n_samples = config["n_samples"])
          val_loss.append(mean_loss_val)

          metrics = {"mean_valid_loss" : mean_loss_val, "AUC_PR" : pr_auc_val, "AUC" : auc_val, "Balanced Accuracy" : balanced_acc_val, "val_recall" : recall_val, "val_precision" : precision_val}

        elif mode == "regression":

          mean_loss_val, var_exp_val, mad_exp_val, r_score_val = validate_bnam(model = model, device = device, mode = mode,
                                            val_loader = valset, loss_fun = loss_fun, kl_weight = config["kl_weight"], batch_size = config["batch_size"], n_samples = config["n_samples"])

          val_loss.append(mean_loss_val)

          metrics =  {"mean_valid_loss" : mean_loss_val, "val_accuracy" : var_exp_val, "val_recall" : mad_exp_val, "val_precision" : r_score_val}

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
            tune.report(metrics, checkpoint=checkpoint)

    print("Finished Training!")


# train BNAIM

def validate_bnaim(model, device, mode, val_loader, loss_fun, kl_weight, batch_size, n_samples):

    val_loss = []

    target_lis = []
    pred_lis = []
    norm_pred = []

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

            mean_pred = mean_pred.squeeze(-1)
            log_lik_loss = loss_fun(mean_pred, target)

            scaled_kl = kl_loss * kl_weight / batch_size

            loss = log_lik_loss + scaled_kl
            val_loss.append(loss.cpu())

            target_lis.append(target.detach().cpu())
            norm_pred.append(torch.sigmoid(mean_pred).detach().cpu())


            pred_lis.append(torch.round(torch.sigmoid(mean_pred)).detach().cpu() if mode == "classification" else
                            mean_pred.detach().cpu())

        mean_loss = np.mean(np.array(val_loss))

        target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)
        norm_pred_lis = torch.cat(norm_pred)


        if mode == "classification":
          pre, re, _ = precision_recall_curve(target_ten, norm_pred_lis) #other paper uses roc_curve here
          auc_precision_recall = auc(re, pre)

          pre_2, re_2, _ = roc_curve(target_ten, norm_pred_lis)
          auc_val = auc(pre_2, re_2)

          balanced_acc = balanced_accuracy_score(target_ten, pred_ten)
          recall = recall_score(target_ten, pred_ten)
          precision = precision_score(target_ten, pred_ten)


          return mean_loss, auc_precision_recall, auc_val, balanced_acc, recall, precision
        
        else:
           var_exp = var_exp_score(pred_ten, target_ten)
           mad_exp = mad_explained(pred_ten, target_ten)
           r_score = coef_det(pred_ten, target_ten)

           return mean_loss, var_exp, mad_exp, r_score



def train_bnaim(config):

    root_dir = 'data/cxr_images_ny'
    data_dir = 'data/patient_data.csv'

    device = config["device"]
    mode = config["mode"]
    target = config["target"]
  
    trainset, valset, _ , features = dataloaders_img(target_column = target, train_frac = 0.7, val_frac = 0.2, batch_size = config["batch_size"],
                                                     root_dir = root_dir, data_dir=data_dir)

    model = make_model(config, features.shape[-1])

    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])

    if mode == "classification":
      loss_fun = F.binary_cross_entropy_with_logits

    elif mode == "regression": 
      loss_fun = nn.MSELoss()

    loss_lis = []
    overall_loss = []
    val_loss = []

    model = model.to(device)
    model.train()

    for epoch in range(config["n_epochs"]):
        target_lis = []
        pred_lis = []

        for i, batch in enumerate(tqdm(trainset)):

            images, features, target = batch

            images = images.to(device)
            target = target.to(device)
            features = features.to(device)

            output = []
            kl_div = []

            for _ in range(config["n_samples"]):
                out, kl = model(images, features)
                output.append(out)
                kl_div.append(kl)


            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)


            mean_pred = mean_pred.squeeze(-1)

            loss = loss_fun(mean_pred, target)
            scaled_kl = kl_loss * config["kl_weight"] / config["batch_size"]
            loss += scaled_kl  #ELBO Loss add if loos_fun is negative log_likelihood

        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_lis.append(loss.cpu().detach())
            target_lis.append(target.detach().cpu())
            pred_lis.append(torch.round(torch.sigmoid(mean_pred)).detach().cpu() if mode == "classification" else
                            mean_pred.detach().cpu())


        mean_loss = np.mean(np.array(loss_lis))
        overall_loss.append(mean_loss)
        loss_lis = []

        target_ten, pred_ten = torch.cat(target_lis), torch.cat(pred_lis)

        if mode == "classification":
          
          mean_loss_val, pr_auc_val, auc_val, balanced_acc_val, recall_val, precision_val = validate_bnaim(model = model, device = device, mode = mode,
                                          val_loader = valset, loss_fun = loss_fun, kl_weight = config["kl_weight"], batch_size = config["batch_size"], n_samples = config["n_samples"])
          val_loss.append(mean_loss_val)
      
          metrics = {"mean_valid_loss" : mean_loss_val, "AUC_PR" : pr_auc_val, "AUC" : auc_val, "Balanced Accuracy" : balanced_acc_val, "val_recall" : recall_val, "val_precision" : precision_val}

        elif mode == "regression":

          mean_loss_val, var_exp_val, mad_exp_val, r_score_val = validate_bnaim(model = model, device = device, mode = mode,
                                            val_loader = valset, loss_fun = loss_fun, kl_weight = config["kl_weight"], batch_size = config["batch_size"], n_samples = config["n_samples"])

          val_loss.append(mean_loss_val)

          metrics =  {"mean_valid_loss" : mean_loss_val, "val_accuracy" : var_exp_val, "val_recall" : mad_exp_val, "val_precision" : r_score_val}

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
            tune.report(metrics, checkpoint=checkpoint)

    print("Finished Training!")


def get_test_predictions(best_result):
    
    mode = best_result.config["mode"]
    device = best_result.config["device"]
    target = best_result.config["target"]
    

    _, _, test_loader, features = dataloaders_img(target_column = target, train_frac = 0.7, val_frac = 0.2, batch_size = best_result.config["batch_size"])

    best_trained_model = make_model(best_result.config, features.shape[-1])
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    print(checkpoint_path)

    model_state, _optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    target_lis = []
    pred_lis = []
    norm_pred = []

    best_trained_model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            images, features, target = batch

            images = images.to(device)
            target = target.to(device)
            features = features.to(device)

            output = []

            for sample in range(best_result.config["n_post_samples"]):
                out, _ = best_trained_model(images, features)
                output.append(out)

            mean_pred = torch.mean(torch.stack(output), dim = 0)

            norm_pred.append(torch.sigmoid(mean_pred).detach().cpu())

            target_lis.append(target.detach().cpu())

            pred_lis.append(torch.round(torch.sigmoid(mean_pred)).detach().cpu() if mode == "classification" else
                            mean_pred.detach().cpu())

        target_ten, pred_ten, norm_pred_lis = torch.cat(target_lis), torch.cat(pred_lis), torch.cat(norm_pred)

        if mode == "classification":
          pre, re, thresholds = precision_recall_curve(target_ten, norm_pred_lis) #other paper uses roc_curve here
          auc_precision_recall = auc(re, pre)

          pre_2, re_2, thresholds_2 = roc_curve(target_ten, norm_pred_lis)
          auc_val = auc(pre_2, re_2)

          balanced_acc = balanced_accuracy_score(target_ten, pred_ten)
          recall = recall_score(target_ten, pred_ten)
          precision = precision_score(target_ten, pred_ten)

          results = {"AUC_PR" : auc_precision_recall, "AUC" : auc_val, "Balanced Accuracy" : balanced_acc, "Recall" : recall, "Precision" : precision}
          
          with open(f'test_results_{best_result.config["name"]}_{best_result.config["target"]}.json', "w") as f:
            json.dump(results, f, indent=4)

          return auc_precision_recall, auc_val, balanced_acc, recall, precision
        else:
           var_exp = var_exp_score(pred_ten, target_ten)
           mad_exp = mad_explained(pred_ten, target_ten)
           r_score = coef_det(pred_ten, target_ten)

           return var_exp, mad_exp, r_score


def test_predictions_bnaim(config, ckpt_path, target):

    mode = config["mode"]
    device = config["device"]
    target = config["target"]


    _, _, test_loader, features = dataloaders_img(target_column = target, train_frac = 0.7, val_frac = 0.2, batch_size = config["batch_size"],
                                                  root_dir = '//content/drive/MyDrive/bayesNAIM/cxr_images_ny/', data_dir = '/content/drive/MyDrive/bayesNAIM/patient_data_raw.csv')

    best_trained_model = make_model(config, features.shape[-1])
    best_trained_model.to(device)

    checkpoint_path = ckpt_path

    model_state, _optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    target_lis = []
    pred_lis = []
    norm_pred = []

    best_trained_model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            images, features, target = batch

            images = images.to(device)
            target = target.to(device)
            features = features.to(device)

            output = []

            for sample in range(100):
                out, _ = best_trained_model(images, features)
                output.append(out)

            mean_pred = torch.mean(torch.stack(output), dim = 0)

            norm_pred.append(torch.sigmoid(mean_pred).detach().cpu())

            target_lis.append(target.detach().cpu())

            pred_lis.append(torch.round(torch.sigmoid(mean_pred)).detach().cpu() if mode == "classification" else
                            mean_pred.detach().cpu())

        target_ten, pred_ten, norm_pred_lis = torch.cat(target_lis), torch.cat(pred_lis), torch.cat(norm_pred)

        if mode == "classification":
          pre, re, thresholds = precision_recall_curve(target_ten, norm_pred_lis) #other paper uses roc_curve here
          auc_precision_recall = auc(re, pre)

          pre_2, re_2, thresholds_2 = roc_curve(target_ten, norm_pred_lis)
          auc_val = auc(pre_2, re_2)

          balanced_acc = balanced_accuracy_score(target_ten, pred_ten)
          recall = recall_score(target_ten, pred_ten)
          precision = precision_score(target_ten, pred_ten)

          results = {"AUC_PR" : auc_precision_recall, "AUC" : auc_val, "Balanced Accuracy" : balanced_acc, "Recall" : recall, "Precision" : precision}

          with open(f'test_results_{config["name"]}_{config["target"]}.json', "w") as f:
            json.dump(results, f, indent=4)

          return results, pre, re, target_ten, norm_pred_lis
        else:
           var_exp = var_exp_score(pred_ten, target_ten)
           mad_exp = mad_explained(pred_ten, target_ten)
           r_score = coef_det(pred_ten, target_ten)

           return var_exp, mad_exp, r_score
        


def test_predictions_bnam(config, ckpt_path):

    mode = config["mode"]
    device = config["device"]
    target = config["target"]


    _, _, test_loader, features = dataloaders(target_column = target, train_frac = 0.7, val_frac = 0.2, batch_size = config["batch_size"],
                                             root_dir = '//content/drive/MyDrive/bayesNAIM/cxr_images_ny/', data_dir = '/content/drive/MyDrive/bayesNAIM/patient_data_raw.csv')

    best_trained_model = make_model(config, features.shape[-1])
    best_trained_model.to(device)

    checkpoint_path = ckpt_path

    model_state = torch.load(checkpoint_path) #model_state, _optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    target_lis = []
    pred_lis = []
    norm_pred = []

    best_trained_model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            features, target = batch
            target = target.to(device)
            features = features.to(device)

            output = []

            for sample in range(100):
                out, _ = best_trained_model(features)
                output.append(out)

            mean_pred = torch.mean(torch.stack(output), dim = 0)

            norm_pred.append(torch.sigmoid(mean_pred).detach().cpu())

            target_lis.append(target.detach().cpu())

            pred_lis.append(torch.round(torch.sigmoid(mean_pred)).detach().cpu() if mode == "classification" else
                            mean_pred.detach().cpu())

        target_ten, pred_ten, norm_pred_lis = torch.cat(target_lis), torch.cat(pred_lis), torch.cat(norm_pred)

        if mode == "classification":
          pre, re, thresholds = precision_recall_curve(target_ten, norm_pred_lis) #other paper uses roc_curve here
          auc_precision_recall = auc(re, pre)

          pre_2, re_2, thresholds_2 = roc_curve(target_ten, norm_pred_lis)
          auc_val = auc(pre_2, re_2)

          balanced_acc = balanced_accuracy_score(target_ten, pred_ten)
          recall = recall_score(target_ten, pred_ten)
          precision = precision_score(target_ten, pred_ten)

          results = {"AUC_PR" : auc_precision_recall, "AUC" : auc_val, "Balanced Accuracy" : balanced_acc, "Recall" : recall, "Precision" : precision}

          with open(f'test_results_{config["name"]}_{config["target"]}.json', "w") as f:
            json.dump(results, f, indent=4)

          return results, pre, re, target_ten, norm_pred_lis
        else:
           var_exp = var_exp_score(pred_ten, target_ten)
           mad_exp = mad_explained(pred_ten, target_ten)
           r_score = coef_det(pred_ten, target_ten)

           return var_exp, mad_exp, r_score
        


def test_predictions_nam(config, ckpt_path):

    mode = config["mode"]
    device = config["device"]
    target = config["target"]


    _, _, test_loader, features = dataloaders(target_column = target, train_frac = 0.7, val_frac = 0.2, batch_size = config["batch_size"],
                                                  root_dir = '//content/drive/MyDrive/bayesNAIM/cxr_images_ny/', data_dir = '/content/drive/MyDrive/bayesNAIM/patient_data_raw.csv')

    best_trained_model = make_model(config, features.shape[-1])
    best_trained_model.to(device)

    checkpoint_path = ckpt_path

    model_state, _optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    target_lis = []
    pred_lis = []
    norm_pred = []

    best_trained_model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            features, target = batch

            target = target.to(device)
            features = features.to(device)

            out = best_trained_model(features)

            out = out.squeeze(-1)

            norm_pred.append(torch.sigmoid(out).detach().cpu())

            target_lis.append(target.detach().cpu())

            pred_lis.append(torch.round(torch.sigmoid(out)).detach().cpu() if mode == "classification" else
                            mean_pred.detach().cpu())

        target_ten, pred_ten, norm_pred_lis = torch.cat(target_lis), torch.cat(pred_lis), torch.cat(norm_pred)

        if mode == "classification":
          pre, re, thresholds = precision_recall_curve(target_ten, norm_pred_lis) #other paper uses roc_curve here
          auc_precision_recall = auc(re, pre)

          pre_2, re_2, thresholds_2 = roc_curve(target_ten, norm_pred_lis)
          auc_val = auc(pre_2, re_2)

          balanced_acc = balanced_accuracy_score(target_ten, pred_ten)
          recall = recall_score(target_ten, pred_ten)
          precision = precision_score(target_ten, pred_ten)

          results = {"AUC_PR" : auc_precision_recall, "AUC" : auc_val, "Balanced Accuracy" : balanced_acc, "Recall" : recall, "Precision" : precision}

          with open(f'test_results_{config["name"]}_{config["target"]}.json', "w") as f:
            json.dump(results, f, indent=4)

          return results, pre, re, target_ten, norm_pred_lis
        else:
           var_exp = var_exp_score(pred_ten, target_ten)
           mad_exp = mad_explained(pred_ten, target_ten)
           r_score = coef_det(pred_ten, target_ten)

           return var_exp, mad_exp, r_score
