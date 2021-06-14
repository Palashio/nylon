from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_numeric_dtype
import os
from sklearn.ensemble import BaggingClassifier
import sys
from importlib import import_module
import shutil
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
import ssl
from sklearn.model_selection import cross_val_score
from nylon.preprocessing.preprocessing import (initial_preprocessor, structured_preprocessor)
import nltk
from nylon.preprocessing.preprocessing import process_dates
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller
from nylon.modeling.modeling import (a_svm, nearest_neighbors, a_tree, sgd, gradient_boosting, adaboost, rf, mlp, default_modeling, svm_stroke, ensemble_stroke)
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer, LabelEncoder)


preprocess_vocab = {'one-hot', 'label-encode', 'fill', 'scale', 'dates', 'custom', 'min-max', 'ordinal', 'importance', 'clean-text', 'embed'}
modeling_vocab = {'linear', 'svm', 'decision', 'sgd', 'neighbors', 'adaboost', 'gradient-boost', 'rf', 'svms', 'ensembles'}
analysis_vocab = ['cross-val', 'acc-score', 'confusion', 'pr', 'importances', 'ALL']
models = {"gradient-boost": gradient_boosting, "svm" : a_svm, "neighbors" : nearest_neighbors, "decision" : a_tree, "sgd" : sgd, "adaboost" : adaboost, 'rf' : rf, 'mlp' : mlp, 'svms' : svm_stroke, 'ensembles' : ensemble_stroke }
just_models = {"svm", "neighbors", "decision", "sgd", "adaboost", "rf", "mlp"}

def preprocess_module(request_info):
    json_file = request_info['json']
    df = request_info['df']

    if 'preprocessor' in json_file:
        preprocess = json_file['preprocessor']
        for element in preprocess:
            if element not in preprocess_vocab:
                raise Exception("Your specificed preprocessing technique -- {} -- is not supported".format(element))
            if element == "custom":

                sys_path = "/nylon/supplementaries/buffer/"
                sys.path.insert(1, os.getcwd() + sys_path)

                absolute_path = os.path.abspath(os.getcwd()) + '/nylon/supplementaries/buffer/temp.py'

                file_name = json_file['preprocessor']['custom']['loc'].rsplit("/")[-1]
                shutil.copy(json_file['preprocessor']['custom']['loc'], absolute_path)

                mod = import_module('temp')
                new_func = getattr(mod, json_file['preprocessor']['custom']['name'])

                df = new_func(df, json_file)
                sys.path.remove(sys_path)
                os.remove("./buffer/temp.py")

            if element == "label-encode":
                columns = preprocess['label-encode']

                if isinstance(columns, str):
                    columns = [columns]

                for column in columns:
                    enc = LabelEncoder()
                    resulting_encoder = enc.fit_transform(df[column])
                    df = df.assign(ocean_proximity=resulting_encoder)
                    # df1 = df1.assign(e=e.values)
            if element == 'dates':
                columns = preprocess['dates']

                if isinstance(columns, str):
                    columns = [columns]

                df = process_dates(df, columns)

            if element == 'clean-text':
                columns = preprocess['clean-text']
                if isinstance(columns, str):
                    columns = [columns]

                df = text_preprocessing(df, columns)
            if element == 'embed':
                columns = preprocess['embed']
                if isinstance(columns, str):
                    columns = [columns]

                df = embedding_preprocessor(df, columns)
            # if element == 'tokenize':
            #     columns = preprocess['tokenize']
            #
            #     if isinstance(columns, str):
            #         columns = [columns]
            #
            #     df = tokenize_text(df, columns)

            if element == "fill":
                if preprocess['fill'] == 'ALL':
                    df = df.dropna()
                else:
                    if 'column' not in preprocess['fill']:
                        raise Exception("A target column should be specified with the -- column -- command.")
                    if 'target' not in preprocess['fill']:
                        target = 'nan'
                    else:
                        target = preprocess['fill']['target']

                    if 'tactic' not in preprocess['fill']:
                        tactic = 'mean'
                    else:
                        tactic = preprocess['fill']['tactic']

                    columns = preprocess['fill']['column']
                    if isinstance(columns, str):
                        columns = [columns]

                    for column in columns:
                        column_transformed = np.array(df[column]).reshape(-1, 1)
                        imputer = SimpleImputer(missing_values=(np.nan if target == 'nan' else target),
                                                    strategy=tactic)

                        df[column] = imputer.fit_transform(np.array(df[column]).reshape(-1, 1))

            if element == "scale":
                columns = preprocess["scale"]

                if isinstance(columns, str):
                    columns = [columns]

                for column in columns:
                    if column not in df.columns:
                        raise Exception(
                                "The target column you provided -- {} -- does not exist".format(preprocess['scale']))
                    if not is_numeric_dtype(df[column]):
                            raise Exception("You can only scale numeric columns, your column was a boolean.")

                    scaler = StandardScaler()
                    df[column] = scaler.fit_transform(np.array(df[column]).reshape(-1, 1))
            if element == 'min-max':
                columns = preprocess["min-max"]

                if isinstance(columns, str):
                    columns = [columns]
                for column in columns:
                    if column not in df.columns:
                        raise Exception(
                                "The target column you provided -- {} -- does not exist".format(preprocess['min-max']))
                    if not is_numeric_dtype(df[column]):
                        raise Exception("You can only scale numeric columns, your column was a boolean.")

                    scaler = MinMaxScaler()
                    df[column] = scaler.fit_transform(np.array(df[column]).reshape(-1, 1))

            if element == "ordinal":
                if element == "ordinal":
                    columns = preprocess['ordinal']

                    if isinstance(columns, str):
                        columns = [columns]

                    for column in columns:
                        enc = OrdinalEncoder()
                        df[column] = enc.fit_transform(np.array(df[column]).reshape(-1, 1))
            if element == "importance":
                number = preprocess['importance']

                if not isinstance(number, int):
                    raise Exception("Please provide the number of columns you'd like to remove based on importance")

                forest = RandomForestRegressor()

                target = json_file['data']['target']
                y = df[target]
                del df[target]

                forest.fit(df, y)
                importance = forest.feature_importances_

                if number < 0:
                    lowest_indices = importance.argsort()[:number * -1]

                    for index, column in enumerate(df.columns):
                        if index in lowest_indices:
                            del df[column]
                else:
                    top_indices = np.argpartition(importance, number*-1)[number * 1:]

                    for index, column in enumerate(df.columns):
                        if index not in top_indices:
                            del df[column]

                df[target] = y.values

            if element == "one-hot":
                columns = preprocess['one-hot']

                if isinstance(columns, str):
                    columns = [columns]

                for column in columns:
                    if is_numeric_dtype(column):
                            raise Exception("You can only one hot encode on numeric columns, your column was a boolean.")
                    if column == json_file['data']['target']:
                        raise Exception("You cannot one-hot-encode the target column you've specified")

                    label = LabelEncoder()
                    df[column] = label.fit_transform(df[preprocess['one-hot']])

                    enc = OneHotEncoder()
                    columns = [column + str(i) for i in range(len(np.unique(df[column])))]
                    enc_df = pd.DataFrame(enc.fit_transform(df[[column]]).toarray(), columns=columns)

                    for value in enc_df.columns:
                        df[value] = enc_df[value].values

                    del df[column]

        target = json_file['data']['target']
        y = df[target]

        del df[target]
        X_train, X_test, y_train, y_test = train_test_split(
                df, y, test_size=0.2)


        df = {
                'train': pd.concat([X_train], axis=1),
                'test': pd.concat([X_test], axis=1)
            }
        y = {'train': y_train, 'test': y_test}

        try:
            val = LinearRegression().fit(df['train'], y['train'])
        except:
            raise Exception("The preprocessing modules you provided is not sufficient")
    else:
        df, y = initial_preprocessor(df, json_file)

    request_info['df'] = df
    request_info['y'] = y

    return request_info


def modeling_module(request_info):
    json_file = request_info['json']
    df = request_info['df']
    y = request_info['y']


    if 'modeling' not in json_file:
        model = default_modeling(df, y)
    else:
        if 'custom' in json_file['modeling']:

            sys_path = "/nylon/supplementaries/buffer/"
            sys.path.insert(1, os.getcwd() + sys_path)

            absolute_path = os.path.abspath(os.getcwd()) + '/nylon/supplementaries/buffer/temp.py'
            file_name = json_file['modeling']['custom']['loc'].rsplit("/")[-1]
            shutil.copy(json_file['modeling']['custom']['loc'], absolute_path)

            mod = import_module('temp')
            new_func = getattr(mod, json_file['modeling']['custom']['name'])

            model = new_func(df['train'], y['train'])
            sys.path.remove(sys_path)
            os.remove("./buffer/temp.py")

        if 'type' in json_file['modeling']:
            type_model = json_file['modeling']['type']
            model_storage = []
            accs = []

            if isinstance(type_model, str):
                type_model = [type_model]


            fifty_train, _, fifty_y, _ = train_test_split(df['train'], y['train'], test_size=0.5)

            for a_model in type_model:
                if a_model not in modeling_vocab:
                    raise Exception(
                        "The specified model -- {} -- is not in the list of available models".format(
                            a_model))

                model = models[a_model](fifty_train, fifty_y, json_file=json_file['modeling'])
                model_storage.append(model)

                curr_acc = accuracy_score(model.predict(df['test']), y['test'])
                accs.append(curr_acc)

            max_accuracy = accs.index(max(accs))
            best_model_type = type_model[max_accuracy]
            model = models[best_model_type](df['train'], y['train'], json_file=json_file['modeling'])
        elif 'voting' in json_file['modeling']:

            classifiers = json_file['modeling']['voting']

            if len(classifiers) < 2:
                raise Exception("Voting classifiers cannot have less than two models for the ensemble.")

            voting_classifier = VotingClassifier(estimators=return_models_voting(classifiers))
            voting_classifier.fit(df['train'], y['train'])

            model = voting_classifier

        elif 'bag' in json_file['modeling']:
            classifier = json_file['modeling']['bag']

            if isinstance(classifier, str):
                pass
            else:
                raise Exception("You can only use one base classifier for bagging.")

            bagging_classifier = BaggingClassifier(base_estimator=classifier)

            bagging_classifier.fit(df['train'], y['train'])
            model = bagging_classifier

    request_info['df'] = df
    request_info['model'] = model
    request_info['y'] = y

    return request_info


def return_models_voting(classifier_list):
    estimators = []

    for classifier in classifier_list:
        clf = models[classifier](None, None, {"no_params": 0}, trained=False)
        estimators.append((str(clf), clf))

    return estimators



def analysis_module(request_info):
    json_file = request_info['json']
    df = request_info['df']
    model = request_info['model']
    y = request_info['y']

    analysis_tuple = {}

    cv_results = cross_score(json_file, df, model, y)
    analysis_tuple['cross-validation'] = cv_results

    if 'analysis' in json_file:
        analysis = json_file['analysis']
    else:
        json_file['analysis'] = {'type': 'ALL'}
    if True:
        for element in json_file['analysis']:
            if element == "type":
                analysis_group = json_file['analysis']['type']
                if isinstance(analysis_group, str):
                    analysis_group = [analysis_group]
                for analysis_type in analysis_group:
                    if analysis_type not in analysis_vocab:
                        raise Exception("Your specific type of analysis -- {} -- is not in the supported formats".format(analysis_type))
                    if analysis_type == 'acc-score':
                        acc_results = acc_score(model, df, y)
                        analysis_tuple['acc-score'] = acc_results
                    if analysis_type == 'confusion':
                        matrix = confusion(model, df, y)
                        analysis_tuple['confusion_matrix'] = matrix
                    if analysis_type == 'pr':
                        analysis_tuple['precision'] = precision_calculation(model, df, y)
                        analysis_tuple['recall'] = recalll(model, df, y)
                    if analysis_type == 'importances':
                        analysis_tuple['importances'] = feature_importances(model, df, y)
                    if analysis_type == 'ALL':
                        acc, cv, matrix, precision, recaller = default_analysis(json_file, model, df, y)
                        analysis_tuple['acc-score'] = acc
                        analysis_tuple['cross-validation'] = cv
                        analysis_tuple['confusion_matrix'] = matrix
                        analysis_tuple['precision'] = precision
                        analysis_tuple['recall'] = recaller
            if element == "custom":
                sys_path = "/nylon/supplementaries/buffer/"
                sys.path.insert(1, os.getcwd() + sys_path)

                absolute_path = os.path.abspath(os.getcwd()) + '/nylon/supplementaries/buffer/temp.py'

                file_name = json_file['analysis']['custom']['loc'].rsplit("/")[-1]
                shutil.copy(json_file['analysis']['custom']['loc'], absolute_path)

                mod = import_module('temp')
                new_func = getattr(mod, json_file['analysis']['custom']['name'])

                results, name = new_func(json_file, df, model, y)
                sys.path.remove(sys_path)
                os.remove("./buffer/temp.py")

                analysis_tuple[name] = results


    request_info['analysis'] = analysis_tuple
    return request_info

def default_analysis(json_file, model, df, y):
    acc_results = acc_score(model, df, y)
    cv_results = cross_score(json_file, df, model, y)
    matrix = confusion(model, df, y)
    precise = precision_calculation(model, df, y)
    recaller = recalll(model, df, y)
    return acc_results, cv_results, matrix, precise, recaller


def acc_score(model, df, y):
    return accuracy_score(y['test'], model.predict(df['test']))

def cross_score(json_file, df, model, y):
    (y)
    if 'analysis' in json_file:
        cv_fold = (json_file['analysis']['spec'] if 'spec' in json_file['analysis'] else 5)
    else:
        cv_fold = 2
    return cross_val_score(model, df['train'], y['train'], cv=cv_fold).tolist()

def confusion(model, df, y):
    unpacked_matrix = {}
    matrix = confusion_matrix(model.predict(df['test']), y['test']).tolist()
    for i, row in enumerate(matrix):
        unpacked_matrix[str(i + 1)] = list(row)

    return unpacked_matrix

def precision_calculation(model, df, y):
    output = model.predict(df['test'])
    average_value = ('binary' if len(np.unique(y['test'])) == 2 else 'macro')


    return float(precision_score(y['test'], output, average=average_value, labels=np.unique(output)))

def recalll(model, df, y):
    output = model.predict(df['test'])
    average_value = ('binary' if len(np.unique(y['test'])) == 2 else 'macro')
    return float(recall_score(y['test'], output, average=average_value, labels=np.unique(output)))


def text_preprocessing(combined, text_cols):
    nltk_downloads()
    lemmatizer = WordNetLemmatizer()
    # combined = pd.concat([data['train'], data['test']], axis=0)

    spell = Speller(fast=True)
    for col in text_cols:
        combined[col] = combined[col].apply(
            lambda x: x.lower() if isinstance(x, str) else x)

    stop_words = set(stopwords.words('english'))

    for col in text_cols:
        preprocessed_text = []
        for words in combined[col]:
            if words is not np.nan:
                words = word_tokenize(words)
                words = [word for word in words if word.isalpha()]
                words = [word for word in words if word not in stop_words]
                words = [spell(word) for word in words]
                words = [lemmatizer.lemmatize(word) for word in words]

                preprocessed_text.append(' '.join(words))

            else:
                preprocessed_text.append(np.nan)

        combined[col] = preprocessed_text

    return combined


def nltk_downloads():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

def feature_importances(model, df, y):
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(df['train'], y['train'])
    importances = rf.feature_importances_
    
    importance_dict = {}
    
    for i, column in enumerate(df['train'].columns):
        importance_dict[column] = importances[i]


    return importance_dict

# def tokenize_text(dataset, columns):
#     nlp = English()
#     # Create a Tokenizer with the default settings for English
#     # including punctfuation rules and exceptions
#     tokenizer = nlp.Defaults.create_tokenizer(nlp)
#
#     for column in columns:
#         df_column = dataset[column]
#         (df_column)
#         for i, value in enumerate(df_column):
#             df_column
#         # for value in dataset[column]:
#         #     dataset[column] = tokenizer(dataset[column])
#         #     #(value)
#
#     return dataset

def embedding_preprocessor(data, text):
    full_pipeline = ColumnTransformer([], remainder="passthrough")
    if len(text) != 0:
        # Each text col needs a separate pipeline
        for x in range(len(text)):
            full_pipeline.transformers.append(
                (f"text_{x}",
                 Pipeline(
                     [
                         ('test',
                          FunctionTransformer(
                              lambda x: np.reshape(
                                  x.to_numpy(),
                                  (-1,
                                   1)))),
                         ('imputer',
                          SimpleImputer(
                              strategy="constant",
                              fill_value="")),
                         ('raveler',
                          FunctionTransformer(
                              lambda x: x.ravel(),
                              accept_sparse=True)),
                         ('vect',
                          TfidfVectorizer()),
                         ('densifier',
                          FunctionTransformer(
                              lambda x: x.todense(),
                              accept_sparse=True)),
                         ('embedder',
                          FunctionTransformer(
                              textembedder,
                              accept_sparse=True))]),
                    text[x]))


    data_dict = {}
    data_dict['train'] = data
    train = full_pipeline.fit_transform(data_dict['train'])
    # test = full_pipeline.transform(data['test'])


    # Ternary clause because when running housing.csv,
    # the product of preprocessing is np array, but not when using landslide
    # data... not sure why
    final_data = pd.DataFrame(
        (train.toarray() if not isinstance(
            train,
            np.ndarray) else train),
        columns=data.columns)

    return final_data

def textembedder(text):

    total = list()
    for i in text:
        total.append(np.sum(i))

    return np.reshape(total, (-1, 1))




