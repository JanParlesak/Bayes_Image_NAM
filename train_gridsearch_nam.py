from train import *
from helpers_data import * 
import tempfile


from ray import tune
from ray.tune.schedulers import ASHAScheduler



config_last_status = {
    "name": 'nam',
    "mode": 'classification',
    "dropout_rate": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    "feature_dropout_rate": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "learning_rate": tune.choice([0.01, 0.001, 0.0001]),
    "n_epochs" : 120,
    "n_samples": 1,
    "n_post_samples": 1,
    "kl_weight" : 0.01,
    "num_trials": 25,
    "target": 'last.status',
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

config_was_ventilated = {
    "name": 'nam',
    "mode": 'classification',
    "dropout_rate": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    "feature_dropout_rate": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "prior_scale" : 0.1,
    "learning_rate": tune.choice([0.01, 0.001, 0.0001]),
    "n_epochs" : 120,
    "n_samples": 1,
    "n_post_samples": 10,
    "kl_weight" : 0.01,
    "num_trials": 25,
    "target": 'was_ventilated',
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


config_icu = {
    "name": 'nam',
    "mode": 'classification',
    "dropout_rate": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    "feature_dropout_rate": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "prior_scale" : 0.1,
    "learning_rate": tune.choice([0.01, 0.001, 0.0001]),
    "n_epochs" : 120,
    "n_samples": 1,
    "n_post_samples": 10,
    "kl_weight" : 0.01,
    "num_trials": 25,
    "target": 'is_icu',
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}



def main(config, gpus_per_trial=1):
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config["n_epochs"],
        grace_period=20,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_nam),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="AUC_PR",
            mode="max",
            scheduler=scheduler,
            num_samples=config["num_trials"],
        ),
        run_config=tune.RunConfig(
            name="experiment_nam",
            storage_path="/user/jan.parlesak/u24266/repos/Bayes_Image_NAM/test_results",
            checkpoint_config=tune.CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="AUC_PR",
                checkpoint_score_order='max', checkpoint_at_end=False),
        ),

        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result(scope='all')

    print(f"Best trial config: {best_result.config}")
    print(f"Best AUC_PR: {best_result.metrics['AUC_PR']}, Best AUC: {best_result.metrics['AUC']}, Best Accuracy: {best_result.metrics['Balanced Accuracy']}, Best Recall: {best_result.metrics['val_recall']}, Best Precision: {best_result.metrics['val_precision']}")
    
    #get_test_predictions(best_result)




main(config_last_status, gpus_per_trial=1 if torch.cuda.is_available() else 0)
main(config_was_ventilated, gpus_per_trial=1 if torch.cuda.is_available() else 0)
main(config_icu, gpus_per_trial=1 if torch.cuda.is_available() else 0)