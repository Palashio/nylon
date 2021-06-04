import sys
from reader import DataReader
import json
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from importlib import __import__
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import os
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer, LabelEncoder)
from handlers import (preprocess_module, modeling_module, analysis_module)

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


def controller(dataset_path, json_file_path, save_model=False):


    df, json_file = dataset_initializer(dataset_path, json_file_path)

    df, y = preprocess_module(json_file, df)


    df, model, y = modeling_module(json_file, df, y)

    if save_model:
     pickle.dump(model, open('model.sav', 'wb'))

    tuple = analysis_module(json_file, df, model, y)
    #
    # return_dict = {'target': json_file['data']['target']}
    #
    # return_dict['run_id'] = run_id
    # internal_return_dict = return_dict.copy()
    # internal_return_dict['run_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # internal_return_dict['columns'] = columns
    #
    # if 'description' not in json_file:
    #     json_file['description'] = "Default description for user " + str(user_id)
    #
    # internal_return_dict["file"] = json_file
    # return_dict['dataset_name'] = (json_file['data']['loc'] if 'custom' not in json_file['data'] else json_file['data']['custom']['loc'])
    #
    # for result in tuple.keys():
    #     if result != 'confusion_matrix':
    #         return_dict[result] = tuple[result]
    #     else:
    #         return_dict[result] = tuple[result]
    #
    # if tokens == 0:
    #     tokens += 1
    #
    # return_dict["tokens_used"] = tokens
    #
    # return return_dict, 'model.sav', tokens, internal_return_dict


if __name__ == "__main__":
    controller('housing.csv', 'test.json')
