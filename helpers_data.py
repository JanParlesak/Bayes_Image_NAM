import torch
import sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image


OneHotEncoder = sklearn.preprocessing.OneHotEncoder

class CustomPipeline(Pipeline):
  """Custom sklearn Pipeline to transform data."""

  def apply_transformation(self, x):
    """Applies all transforms to the data, without applying last estimator.

    Args:
      x: Iterable data to predict on. Must fulfill input requirements of first
        step of the pipeline.

    Returns:
      xt: Transformed data.
    """
    xt = x
    for _, transform in self.steps[:-1]:
      xt = transform.fit_transform(xt)
    return xt


def load_cxr_data(target_column, data_dir):
  #change this to data/
  df = pd.read_csv(data_dir)

  cols = ["to_patient_id", "age.splits", "gender_concept_name", "smoking_status_v","39156-5_Body mass index (BMI) [Ratio]", "htn_v", "dm_v", "ckd_v","other_lung_disease_v", "malignancies_v", "76282-3_Heart rate.beat-to-beat by EKG",
          "8480-6_Systolic blood pressure", "9279-1_Respiratory rate", "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry",
          "2823-3_Potassium [Moles/volume] in Serum or Plasma", "2524-7_Lactate [Moles/volume] in Serum or Plasma", "1988-5_C reactive protein [Mass/volume] in Serum or Plasma", 
          "2951-2_Sodium [Moles/volume] in Serum or Plasma"] + [target_column]

  df = df[cols]
  
  df["smoking_status_v"] = df["smoking_status_v"].astype(str)
  df["smoking_status_v"] = np.where((df["smoking_status_v"]=='Current') | (df["smoking_status_v"]=='Former'), 1.0, 0.0)
  df["htn_v"] = df["htn_v"].astype(str)
  df["htn_v"] = np.where(df["htn_v"]=='Yes', 1.0, 0.0)
  df["dm_v"] = df["dm_v"].astype(str)
  df["dm_v"] = np.where(df["dm_v"]=='Yes', 1.0, 0.0)
  df["ckd_v"] = df["ckd_v"].astype(str)
  df["ckd_v"] = np.where(df["ckd_v"]=='Yes', 1.0, 0.0)
  df["other_lung_disease_v"] = df["other_lung_disease_v"].astype(str)
  df["other_lung_disease_v"] = np.where(df["other_lung_disease_v"]=='Yes', 1.0, 0.0)
  df["malignancies_v"] = df["malignancies_v"].astype(str)
  df["malignancies_v"] = np.where(df["malignancies_v"]=='Yes', 1.0, 0.0)

  df['39156-5_Body mass index (BMI) [Ratio]'] = df['39156-5_Body mass index (BMI) [Ratio]'].fillna(df['39156-5_Body mass index (BMI) [Ratio]'].mode()[0])
  df['76282-3_Heart rate.beat-to-beat by EKG'] = df['76282-3_Heart rate.beat-to-beat by EKG'].fillna(df['76282-3_Heart rate.beat-to-beat by EKG'].mode()[0])
  df['8480-6_Systolic blood pressure'] = df['8480-6_Systolic blood pressure'].fillna(df['8480-6_Systolic blood pressure'].mode()[0])
  df['9279-1_Respiratory rate'] = df['9279-1_Respiratory rate'].fillna(df['9279-1_Respiratory rate'].mode()[0])
  df['59408-5_Oxygen saturation in Arterial blood by Pulse oximetry'] = df['59408-5_Oxygen saturation in Arterial blood by Pulse oximetry'].fillna(df['59408-5_Oxygen saturation in Arterial blood by Pulse oximetry'].mode()[0])
  df['2823-3_Potassium [Moles/volume] in Serum or Plasma'] = df['2823-3_Potassium [Moles/volume] in Serum or Plasma'].fillna(df['2823-3_Potassium [Moles/volume] in Serum or Plasma'].mode()[0])
  df['2524-7_Lactate [Moles/volume] in Serum or Plasma'] = df['2524-7_Lactate [Moles/volume] in Serum or Plasma'].fillna(df['2524-7_Lactate [Moles/volume] in Serum or Plasma'].mode()[0])
  df['1988-5_C reactive protein [Mass/volume] in Serum or Plasma'] = df['1988-5_C reactive protein [Mass/volume] in Serum or Plasma'].fillna(df['1988-5_C reactive protein [Mass/volume] in Serum or Plasma'].mode()[0])
  df['2951-2_Sodium [Moles/volume] in Serum or Plasma'] = df['2951-2_Sodium [Moles/volume] in Serum or Plasma'].fillna(df['2951-2_Sodium [Moles/volume] in Serum or Plasma'].mode()[0])
 


  if target_column == "last.status":
    df[target_column] = df[target_column].astype(str)
    df[target_column] = np.where(df[target_column]== "deceased", 1.0, 0.0)

  elif target_column == "was_ventilated":
    df[target_column] = df[target_column].astype(str)
    df[target_column] = np.where(df[target_column]== "Yes", 1.0, 0.0)

  elif target_column == "is_icu":
    df[target_column] = df[target_column].astype(str)
    df[target_column] = np.where(df[target_column]== "True", 1.0, 0.0)

  df = df.dropna()

  df["gender_concept_name"] = df["gender_concept_name"].astype(str)
  df["gender_concept_name"] = np.where((df["gender_concept_name"]=='FEMALE'), 1.0, 0.0)

  return df


def preprocess_data_cxr(root_dir, data_dir, target_column):

  features_frame = load_cxr_data(target_column, data_dir)
  features_frame = features_frame.sample(frac=1, random_state = 1).reset_index(drop=True)


  image_dirs = []
  no_path = []

  for i in range(len(features_frame)):
    img_name = os.path.join(root_dir,
                            f'{features_frame.iloc[i, 0]}.png') # this has to be the image name in the csv

    if os.path.exists(img_name):
      image_dirs.append(img_name)

    else:
      no_path.append(i)

  features_frame = features_frame.drop(no_path)

  features_frame = features_frame.drop("to_patient_id", axis = 1)

  targets = np.array(features_frame[target_column])

  features_frame = features_frame.drop(target_column, axis = 1)

  col_min_max = {}
  x = features_frame
  for col in x:
    try:
      unique_vals = x[col].unique()
      col_min_max[col] = (np.min(unique_vals), np.max(unique_vals))
    except:
      col_min_max[col] = (0.0, 1.0)

  column_names = features_frame.columns
  new_column_names = []
  is_categorical = np.array([dt.kind == 'O' for dt in features_frame.dtypes])
  categorical_cols = features_frame.columns.values[is_categorical]
  numerical_cols = features_frame.columns.values[~is_categorical]

  for index, is_cat in enumerate(is_categorical):
    col_name = column_names[index]
    if is_cat:
      new_column_names += [
          '{}: {}'.format(col_name, val) for val in set(features_frame[col_name])
      ]
    else:
      new_column_names.append(col_name)


  cat_ohe_step = (
      'ohe',
      OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
  )

  cat_pipe = Pipeline([cat_ohe_step])
  num_pipe = Pipeline([('identity', FunctionTransformer(validate=True))])
  transformers = [('cat', cat_pipe, categorical_cols),
                  ('num', num_pipe, numerical_cols)]
  column_transform = ColumnTransformer(transformers=transformers)

  pipe = CustomPipeline([('column_transform', column_transform),
                         ('min_max', MinMaxScaler((-1, 1))), ('dummy', None)])
  df = pipe.apply_transformation(features_frame)


  return image_dirs, df.astype('float32'), targets.astype('float32'), col_min_max, new_column_names



transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomRotation(degrees = (-15, 15)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #torchvision.transforms.CenterCrop(image_size)
    ])

transforms_val_test = torchvision.transforms.Compose([
torchvision.transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def dataloaders_img(target_column, train_frac, val_frac, batch_size, root_dir, data_dir):
  
  transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomRotation(degrees = (-15, 15)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #torchvision.transforms.CenterCrop(image_size)
    ])

  transforms_val_test = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  
  imgs_dir, features, targets, col_min_max, new_col_names = preprocess_data_cxr(root_dir= root_dir, data_dir= data_dir,
                          target_column= target_column)
  

  
  
  train_loader_img, val_loader_img, test_loader_img = train_test_split_features_images(image_dir = imgs_dir,
                                                         features = torch.tensor(features),
                                                         targets = torch.tensor(targets),
                                                         train_frac = train_frac,
                                                         val_frac = val_frac,
                                                         batch_size = batch_size,
                                                         transforms_val_test = transforms_val_test,
                                                         transforms_train = transforms_train)
  
  return train_loader_img, val_loader_img, test_loader_img, features
  
  
def dataloaders(target_column, train_frac, val_frac, batch_size, root_dir, data_dir):
  
  imgs_dir, features, targets, col_min_max, new_col_names = preprocess_data_cxr(root_dir= root_dir, data_dir=data_dir,
                          target_column= target_column)
  
  train_loader, val_loader, test_loader = train_test_split_features(features = torch.tensor(features),
                                                         targets = torch.tensor(targets),
                                                         train_frac = train_frac,
                                                         val_frac = val_frac,
                                                         batch_size = batch_size)
  return train_loader, val_loader, test_loader, features



def oversampling(dataset, targets, batch_size):

  unique, counts = np.unique(targets, return_counts=True)
  class_weights = [1.0/c for c in counts]
  sample_weights = [class_weights[int(i)] for i in targets]
  sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

  return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)


class CustomDataset_img_features(torch.utils.data.Dataset):
    """Face Landmarks dataset."""
    def __init__(self, img_dirs, feature_ten, label_ten, transforms=None):

        self.img_dirs = img_dirs
        self.label_ten = label_ten
        self.feature_ten = feature_ten
        self.transforms = transforms

        assert len(img_dirs) == len(label_ten) ==  len(feature_ten), "img_ten and and feature_ten and label ten must have equal size"
    def __len__(self):
      return len(self.img_dirs)

    def __getitem__(self, idx):
      img_name = self.img_dirs[idx]
      image = Image.open(img_name).convert('RGB')
      feat = self.feature_ten[idx]
      y = self.label_ten[idx]

      if self.transforms:
        image = self.transforms(image)

      return image, feat, y

    def set_transforms(self, transforms):
      self.transforms = transforms


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

    train_loader = oversampling(train_set, batch_size = batch_size, targets = train_y)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_test_split_features_images(features, image_dir, targets, train_frac, val_frac, batch_size, transforms_val_test = None, transforms_train = None):
    tot_len = len(features)

    train_max_idx = int(tot_len*train_frac)
    val_max_idx = int(tot_len*val_frac) + train_max_idx

    train_features = features[:train_max_idx]
    train_images = image_dir[:train_max_idx]
    train_y = targets[:train_max_idx]

    val_features = features[train_max_idx:val_max_idx]
    val_images = image_dir[train_max_idx:val_max_idx]
    val_y = targets[train_max_idx:val_max_idx]

    test_features = features[val_max_idx:]
    test_images = image_dir[val_max_idx:]
    test_y = targets[val_max_idx:]

    train_set = CustomDataset_img_features(train_images, train_features, train_y, transforms = transforms_train)
    val_set = CustomDataset_img_features(val_images, val_features, val_y, transforms = transforms_val_test)
    test_set = CustomDataset_img_features(test_images, test_features, test_y, transforms = transforms_val_test)

    #train_loader = oversampling(train_set, batch_size = batch_size, targets = train_y)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)

    return train_loader, val_loader, test_loader






