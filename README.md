# Bayesian Neural Additive Image Model (BayesNAIM)

This repository is the codebase for the Bayesian Neural Additive Image Model (BayesNAIM). The BayesNAIM combines tabular data with image effects, while providing uncertainty estimates. 


### Datasets

To illustrate the model we use the [COVID-19-NY-SBU dataset](https://doi.org/10.7937/TCIA.BBAG-2923) and predict mortality, ventilation and ICU admission.

### Dependencies

See `requirements.txt`

```
uv pip install -r requirements.txt
```
We use Python 3.10. 

### Checkpoints and data

We provide the checkpoints and data at:

[BayesNAIM Checkpoints and Data](https://drive.google.com/file/d/13QHJIdS5z9fCON_iA7NWhqq8Ioah4M-8/view?usp=drive_link)

## Training

For training we provide the diffusion autoencoder checkpoint at:

[Diffae Checkpoint](https://drive.google.com/file/d/1BXSC0hN8iStpH40IeZLsYAbxJn39rM8D/view?usp=drive_link)

The diffae checkpoint should be in a Bayes_Image_NAM/diffae/checkpoints/ directory and the images in the data directory.
It should look like this:

```text
Bayes_Image_NAM/
├── data/
│ ├── cxr_images_ny/
│ └── patient_data.csv
└── diffae/
└── checkpoints/
└── last.ckpt
```


## Demos

- For an example training run please see: [ICU training](https://github.com/JanParlesak/Bayes_NAIM/blob/main/notebooks/Train_BNAIM_ICU.ipynb).

- For the evaluations see: [Evaluations](https://github.com/JanParlesak/Bayes_NAIM/blob/main/notebooks/BayesNAIM_evals.ipynb).





