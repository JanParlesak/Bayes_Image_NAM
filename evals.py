from helpers_data import *
from configs import *
from metrics import *
from helpers import *
import json


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
                            out.detach().cpu())

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
