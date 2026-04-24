from train import *
from helpers_data import * 
import tempfile


from ray import tune
from ray.tune.schedulers import ASHAScheduler


def bnaim(config):

   device = config["device"]
   mode = config["mode"]
   
   pretrained_encoder = load_encoder(device=device)
   pretrained_encoder.ema_model.eval()
   pretrained_encoder.ema_model.to(device)

   train_loader_img, val_loader_img, test_loader_img, features = load_cxr_data_img(target_column = config["target"], train_frac = 0.7, val_frac = 0.2, batch_size = config["batch_size"])

   bayes_mlp = BayesResFeature(n_input = 512, hid_dim = [500,500,500,500]) 
   bayes_nam = BayesNAM(n_features = features.shape[-1], hidden_units = [100, 100, 100], dropout_rate = config["dropout_rate"], feature_dropout_rate = config["feature_dropout_rate"],
                        prior_scale = config["prior_scale"])

   model = BayesImageNAM(pretrained_encoder = pretrained_encoder, bayes_mlp = bayes_mlp, bayes_nam = bayes_nam)

   if mode == "classification":
      loss_function = F.binary_cross_entropy_with_logits
   else: 
      loss_function = nn.MSELoss()
    
   optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

   model_save_name = 'model_one'
   path = 'checkpoints'

   if tune.get_checkpoint():
        loaded_checkpoint = tune.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

   train_loss, val_loss = train_bnaim(model = model, 
                      optimizer = optimizer, 
                      loss_fun = loss_function, 
                      trainset = train_loader_img, 
                      valset= val_loader_img, 
                      device = device, 
                      n_epochs = config["n_epochs"], 
                      n_samples= config["n_samples"],
                      mode = mode,
                      batch_size= config["batch_size"], 
                      kl_weight= 0.01, 
                      early_stopping = True, 
                      n_epochs_early_stopping = 50, 
                      save_path = path, 
                      print_mod = 10)
   
   

   if mode == "classification":
    auc_pr_test, auc_test, balanced_acc_test, recall_test, precision_test = validate_bnaim(model, device, mode, val_loader, loss_fun, kl_weight, batch_size, n_samples)
    
    metrics = {'AUC_PR': auc_pr_test, 'AUC': auc_test, 'Balanced Accuracy' : balanced_acc_test, 'Recall': recall_test, 'Precision': precision_test}
   elif mode == "regression":
    mean_loss, var_exp, mad_exp, r_score = validate_bnaim(model = model, device = device, mode = mode, val_loader = test_loader_img, 
                                                           loss_fun = loss_function , kl_weight = 0.1, batch_size = config["batch_size"], n_samples = config["n_post_samples"])
    metrics= {'mean_loss': mean_loss, 'var_explained': var_exp, 'mad_explained': mad_exp, 'r_score': r_score}

   print("Finished Training!")









