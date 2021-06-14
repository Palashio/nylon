from nylon.data.reader import DataReader
import json
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import shutil
import sys
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   LabelEncoder)
from importlib import import_module
import os

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


def dataset_initializer(request_info):
    dataset = request_info['df']
    json_file_path = request_info['json']

    if isinstance(json_file_path, str):
        json_file = read_json(json_file_path)
    else:
        json_file = json_file_path

    if "custom" not in json_file['data']:
        if "target" not in json_file['data']:
            raise Exception("A target column has to specified under the -- target -- keyword.")
        df = DataReader(json_file, dataset)
        df = df.data_reader()

    else:
        sys_path = "/nylon/supplementaries/buffer/"
        sys.path.insert(1, os.getcwd() + sys_path)

        absolute_path = os.path.abspath(os.getcwd()) + '/nylon/supplementaries/buffer/temp.py'

        file_name = json_file['data']['custom']['loc'].rsplit("/")[-1]
        shutil.copy(json_file['data']['custom']['loc'], absolute_path)

        mod = import_module('temp')
        new_func = getattr(mod, json_file['data']['custom']['name'])

        df = new_func(json_file)
        sys.path.remove(sys_path)
        os.remove("./buffer/temp.py")

    request_info['df'] = df
    request_info['json'] = json_file
    return request_info
