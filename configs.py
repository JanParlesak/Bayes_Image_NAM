config_bnaim_mortality = {
  "batch_size": 32,
  "device": "cuda",
  "dropout_rate": 0.2,
  "feature_dropout_rate": 0.2,
  "kl_weight": 0.01,
  "learning_rate": 0.01,
  "mode": "classification",
  "n_epochs": 120,
  "n_post_samples": 10,
  "n_samples": 1,
  "name": "bnaim",
  "num_trials": 25,
  "prior_scale": 0.1,
   "target": 'last.status'
}

config_mortality_bnam = {
  "batch_size": 64,
  "device": "cuda",
  "dropout_rate": 0.0,
  "feature_dropout_rate": 0.4,
  "kl_weight": 0.01,
  "learning_rate": 0.01,
  "mode": "classification",
  "n_epochs": 120,
  "n_post_samples": 10,
  "n_samples": 1,
  "name": "bnam",
  "num_trials": 25,
  "prior_scale": 0.1,
  "target": "last.status"
}

config_mortality_nam = {
  "batch_size": 32,
  "device": "cuda",
  "dropout_rate": 0.2,
  "feature_dropout_rate": 0.3,
  "kl_weight": 0.01,
  "learning_rate": 0.001,
  "mode": "classification",
  "n_epochs": 120,
  "n_post_samples": 1,
  "n_samples": 1,
  "name": "nam",
  "num_trials": 25,
  "target": "last.status"
}

config_ventilated_bnaim = {
  "batch_size": 128,
  "device": "cuda",
  "dropout_rate": 0.3,
  "feature_dropout_rate": 0.1,
  "kl_weight": 0.01,
  "learning_rate": 0.001,
  "mode": "classification",
  "n_epochs": 120,
  "n_post_samples": 10,
  "n_samples": 1,
  "name": "bnaim",
  "num_trials": 25,
  "prior_scale": 0.1,
  "target": "was_ventilated"
}

config_ventilated_bnam = {
  "batch_size": 128,
  "device": "cuda",
  "dropout_rate": 0.0,
  "feature_dropout_rate": 0.3,
  "kl_weight": 0.01,
  "learning_rate": 0.01,
  "mode": "classification",
  "n_epochs": 120,
  "n_post_samples": 10,
  "n_samples": 1,
  "name": "bnam",
  "num_trials": 25,
  "prior_scale": 0.1,
  "target": "was_ventilated"
}
config_ventilated_nam = {
  "batch_size": 16,
  "device": "cuda",
  "dropout_rate": 0.2,
  "feature_dropout_rate": 0.3,
  "kl_weight": 0.01,
  "learning_rate": 0.01,
  "mode": "classification",
  "n_epochs": 120,
  "n_post_samples": 10,
  "n_samples": 1,
  "name": "nam",
  "num_trials": 25,
  "prior_scale": 0.1,
  "target": "was_ventilated"
}

config_icu_bnaim = {
  "batch_size": 16,
  "device": "cuda",
  "dropout_rate": 0.2,
  "feature_dropout_rate": 0.5,
  "kl_weight": 0.01,
  "learning_rate": 0.001,
  "mode": "classification",
  "n_epochs": 120,
  "n_post_samples": 10,
  "n_samples": 1,
  "name": "bnaim",
  "num_trials": 25,
  "prior_scale": 0.1,
  "target": "is_icu"
}

config_icu_bnam = {
  "batch_size": 128,
  "device": "cuda",
  "dropout_rate": 0.5,
  "feature_dropout_rate": 0.5,
  "kl_weight": 0.01,
  "learning_rate": 0.01,
  "mode": "classification",
  "n_epochs": 120,
  "n_post_samples": 10,
  "n_samples": 1,
  "name": "bnam",
  "num_trials": 25,
  "prior_scale": 0.1,
  "target": "is_icu"
}

config_icu_nam = {
  "batch_size": 128,
  "device": "cuda",
  "dropout_rate": 0.0,
  "feature_dropout_rate": 0.4,
  "kl_weight": 0.01,
  "learning_rate": 0.01,
  "mode": "classification",
  "n_epochs": 120,
  "n_post_samples": 10,
  "n_samples": 1,
  "name": "nam",
  "num_trials": 25,
  "prior_scale": 0.1,
  "target": "is_icu"
}
