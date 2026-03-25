import json
import os
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class BaseConfig:
    def clone(self):
        return deepcopy(self)

    def inherit(self, another):
        """inherit common keys from a given config"""
        common_keys = set(self.__dict__.keys()) & set(another.__dict__.keys())
        for k in common_keys:
            setattr(self, k, getattr(another, k))

    def propagate(self):
        """push down the configuration to all members"""
        for k, v in self.__dict__.items():
            if isinstance(v, BaseConfig):
                v.inherit(self)
                v.propagate()

    def save(self, save_path):
        """save config to json file"""
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        conf = self.as_dict_jsonable()
        with open(save_path, 'w') as f:
            json.dump(conf, f)

    def load(self, load_path):
        """load json config"""
        with open(load_path) as f:
            conf = json.load(f)
        self.from_dict(conf)

    def from_dict(self, dict, strict=False):
        for k, v in dict.items():
            if not hasattr(self, k):
                if strict:
                    raise ValueError(f"loading extra '{k}'")
                else:
                    print(f"loading extra '{k}'")
                    continue
            if isinstance(self.__dict__[k], BaseConfig):
                self.__dict__[k].from_dict(v)
            else:
                self.__dict__[k] = v

    def as_dict_jsonable(self):
        conf = {}
        for k, v in self.__dict__.items():
            if isinstance(v, BaseConfig):
                conf[k] = v.as_dict_jsonable()
            else:
                if jsonable(v):
                    conf[k] = v
                else:
                    # ignore not jsonable
                    pass
        return conf


def jsonable(x):
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False
    

class ConfigTrain(BaseConfig): 
    seed: int = 0
    train_mode = mode 
    n_features, hidden_units, dropout_rate, feature_dropout_rate, batch_size, prior_scale, learning_rate, device, n_epochs, n_samples, n_post_samples