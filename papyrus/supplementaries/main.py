from papyrus.data.reader import DataReader
import json
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from importlib import __import__
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   LabelEncoder)

preprocess_vocab = {'one-hot': OneHotEncoder, 'label-encode': LabelEncoder, 'fill': SimpleImputer,
                    'scale': StandardScaler}
modeling_vocab = {'linear': LinearRegression}
analysis_vocab = {'cross-val': cross_val_score, 'acc-score': accuracy_score}


def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def dataset_initializer(dataset, json_file_path):
    read_in_json = read_json(json_file_path)
    if "target" not in read_in_json['data']:
         raise Exception("A target column has to specified under the -- target -- keyword.")

    if "custom" not in read_in_json['data']:
        df = DataReader(read_in_json, dataset)
        df = df.data_reader()

    # else:
    #     mod = import_module(json_file['data']['custom']['loc'])
    #     new_func = getattr(mod, json_file['data']['custom']['name'])
    #
    #     df = new_func(json_file)
    #     os.remove(json_file['data']['custom']['loc'] + ".py")

    return df, read_in_json


