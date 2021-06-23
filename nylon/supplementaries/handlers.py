from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.ensemble import BaggingClassifier
import sys
from importlib import import_module
import shutil
from sklearn.ensemble import VotingClassifier
import ssl
from nylon.preprocessing.preprocessing import (initial_preprocessor)
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from nylon.modeling.modeling import (a_svm, nearest_neighbors, a_tree, sgd, gradient_boosting, adaboost, rf, mlp, default_modeling, svm_stroke, ensemble_stroke)
from nylon.preprocessing.preprocessing_methods import handle_scaling, handle_min_max, handle_label_encode, handle_ordinal, handle_filling, handle_importance, handle_one_hot, handle_text, handle_embedding, handle_dates
from nylon.analysis.analysis import default_analysis, acc_score, cross_score, confusion, precision_calculation, recall_score_helper

preprocess_vocab = {'one-hot', 'label-encode', 'fill', 'scale', 'dates', 'custom', 'min-max', 'ordinal', 'importance', 'clean-text', 'embed'}
analysis_vocab = ['cross-val', 'acc-score', 'confusion', 'pr', 'importances', 'ALL']

models = {"gradient-boost": gradient_boosting, "svm" : a_svm, "neighbors" : nearest_neighbors, "decision" : a_tree, "sgd" : sgd, "adaboost" : adaboost, 'rf' : rf, 'mlp' : mlp, 'svms' : svm_stroke, 'ensembles' : ensemble_stroke }

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
                df = handle_label_encode(preprocess, df)
            if element == 'dates':
                df = handle_dates(preprocess, df)
            if element == 'clean-text':
                df = handle_text(preprocess, df)
            if element == 'embed':
                df = handle_embedding(preprocess, df)
            if element == "fill":
                df = handle_filling(preprocess, df)
            if element == "scale":
                df = handle_scaling(preprocess, df)
            if element == 'min-max':
                df = handle_min_max(preprocess, df)
            if element == "ordinal":
                df = handle_ordinal(preprocess, df)
            if element == "importance":
                df = handle_importance(preprocess, df, json_file)
            if element == "one-hot":
                df = handle_one_hot(preprocess, df, json_file)

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
                if a_model not in models.keys():
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
                        analysis_tuple['recall'] = recall_score_helper(model, df, y)
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








