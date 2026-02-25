import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from skimage import io, transform

def create_synthetic_table_normal_response(function_list, noise_variance=1, n_datapoints= 10000, x_min=0, x_max=1, equally_spaced=True):
    """
    Create synthetic tabular data with the functions given
    such that the response y is given as y = f_1(x_1) + f_2(x_2) + ... + f_k(x_k) + eps for eps ~ N(0, noise_variance)
    It plots individual graphs for each function in a single row.

    Params:
        function_list: List of functions [f_1, f_2, ..., f_k] to use for the response
        noise_variance: The variance of the added noise
        n_datapoints: Number of samples to generate
        x_min: Minimum x_value for each covariate
        x_max: Maximum x_value for each covariate
        equally_spaced: If True, use equally spaced covariates; otherwise, use random uniform covariates

    Returns:
        X: Matrix containing the training data, has shape n_datapoints x len(function_list)
        y: Response variable, generated as specified
    """
    # Generate covariates (X) based on the specified method
    if equally_spaced:
        X = np.repeat(np.linspace(x_min, x_max, n_datapoints).reshape(-1, 1), len(function_list), axis = 1)
    else:
        X = np.random.uniform(x_min, x_max, size=(n_datapoints, len(function_list)))

    # Initialize an array to store the response variable
    num_functions = len(function_list)
    Y = np.zeros((num_functions, n_datapoints))


    # Create subplots for each function



    # Generate response variable y based on the given functions and noise
    for i, func in enumerate(function_list):
        Y[i] = func(X[:, i])

    Y = Y - np.mean(Y, axis = 1).reshape(-1, 1)

    noise = np.random.normal(0, np.sqrt(noise_variance), size=n_datapoints)
    if num_functions > 1:
      fig, axs = plt.subplots(1, num_functions, figsize=(15, 5))
      for i, func in enumerate(function_list):
          axs[i].scatter(X[:, i], Y[i] + noise, s=5, label=f'f_{i+1} with noise')
          axs[i].scatter(X[:, i], Y[i], s=5, label=f'f_{i+1}', c = "red")
          axs[i].set_xlabel(f'x_{i+1}')
          axs[i].set_ylabel('y')
          axs[i].legend()

    else:
      fig, axs = plt.subplots(1, num_functions, figsize=(5, 5))
      axs.scatter(X[:, 0], Y[0] + noise, s=5, label=f'f_{1}')
      axs.set_xlabel(f'x_{1}')
      axs.set_ylabel('y')
      axs.legend()

    plt.tight_layout()
    plt.show()

    # Add Gaussian noise with the specified variance
    y_mean = np.sum(Y, axis = 0)


    y = y_mean + noise


    #fig, axs = plt.subplots(1, 2, figsize=(10,5))

    #axs[0].scatter(np.mean(X, axis = 1), y_mean, s=5, label=f'final result without noise')

    #axs[1].scatter(np.mean(X, axis = 1), y, s=5, label=f'final result with noise')
    #axs[1].scatter(np.mean(X, axis = 1), y_mean, s=1, c = "red")

    return X, y, Y, noise


class CustomDataset_features(torch.utils.data.Dataset):
    """Face Landmarks dataset."""
    def __init__(self, feature_ten, label_ten, transforms=None):

        self.label_ten = label_ten
        self.feature_ten = feature_ten
        self.transforms = transforms

        assert len(label_ten) ==  len(feature_ten), "img_ten and and feature_ten and label ten must have equal size"

    def __len__(self):
      return len(self.feature_ten)

    def __getitem__(self, idx):
      feat = self.feature_ten[idx]
      y = self.label_ten[idx]

      return feat, y
    
def train_test_split_features(features, targets, train_frac, val_frac, batch_size, transforms_val_test = None, transforms_train = None):

    tot_len = len(features)

    train_max_idx = int(tot_len*train_frac)
    val_max_idx = int(tot_len*val_frac) + train_max_idx

    train_features = features[:train_max_idx]
    train_y = targets[:train_max_idx]

    val_features = features[train_max_idx:val_max_idx]
    val_y = targets[train_max_idx:val_max_idx]

    test_features = features[val_max_idx:]
    test_y = targets[val_max_idx:]

    train_set = CustomDataset_features(train_features, train_y)
    val_set = CustomDataset_features(val_features, val_y)
    test_set = CustomDataset_features(test_features, test_y)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class CustomDataset_img_features(torch.utils.data.Dataset):
    """Face Landmarks dataset."""
    def __init__(self, img_ten, feature_ten, label_ten, transforms=None):

        self.img_ten = img_ten
        self.label_ten = label_ten
        self.feature_ten = feature_ten
        self.transforms = transforms

        assert len(img_ten) == len(label_ten) ==  len(feature_ten), "img_ten and and feature_ten and label ten must have equal size"
    def __len__(self):
      return len(self.img_ten)

    def __getitem__(self, idx):
      x = self.img_ten[idx]
      feat = self.feature_ten[idx]
      y = self.label_ten[idx]

      if not self.transforms == None:
        x = self.transforms(x)

      return x, feat, y

    def set_transforms(self, transforms):
      self.transforms = transforms

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

class MakeDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, target_column, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_column(string): csv_column containing targets
        """
        self.features_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.targets = target_column
    
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.features_frame.iloc[idx, 0]) # this has to be the image name in the csv 
        image = io.imread(img_name)
        features = self.features_frame.iloc[idx, 1:]
        targets = self.features_frame.iloc[idx, self.targets]
        features = features.drop(self.targets, axis = 1)
        features = np.array([features], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'features': features, 'targets':targets}

        if self.transform:
            sample = self.transform(sample)

        return sample



def train_test_split_features_images(features, images, targets, train_frac, val_frac, batch_size, transforms_val_test = None, transforms_train = None):
    tot_len = len(features)

    train_max_idx = int(tot_len*train_frac)
    val_max_idx = int(tot_len*val_frac) + train_max_idx

    train_features = features[:train_max_idx]
    train_images = images[:train_max_idx]
    train_y = targets[:train_max_idx]

    val_features = features[train_max_idx:val_max_idx]
    val_images = images[train_max_idx:val_max_idx]
    val_y = targets[train_max_idx:val_max_idx]

    test_features = features[val_max_idx:]
    test_images = images[val_max_idx:]
    test_y = targets[val_max_idx:]

    train_set = CustomDataset_img_features(train_images, train_features, train_y, transforms = transforms_train)
    val_set = CustomDataset_img_features(val_images, val_features, val_y, transforms = transforms_val_test)
    test_set = CustomDataset_img_features(test_images, test_features, test_y, transforms = transforms_val_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle=True)

    return train_loader, val_loader, test_loader



def var_exp_score(predictions, targets):
  mean_sum_of_squares = torch.mean((predictions - targets)**2)
  variance_targets = torch.var(targets)
  var_exp = 1- mean_sum_of_squares/variance_targets

  return var_exp

def coef_det(x, y):
  sum_x = torch.sum(x)
  sum_y = torch.sum(y)
  n = len(x)

  numerator = n * torch.sum(x * y) - sum_x*sum_x
  denominator = (n * torch.sum(x**2) - sum_x**2)**0.5 * (n * torch.sum(y**2) - sum_y**2)**0.5

  return numerator/denominator


def mad_explained(predictions, targets):
  mean_sum_of_ad = torch.mean(torch.abs(predictions - targets))
  deviation_median = torch.mean(torch.abs(targets - torch.median(targets)))

  mad_exp = 1 - mean_sum_of_ad/deviation_median

  return mad_exp




