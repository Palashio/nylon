from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

import warnings

warnings.filterwarnings('ignore')


def default_modeling(df, y):
    '''
    default modeling function to try all classifiers, and return the best one.
    :param df pandas dataframe object
    :param y is the target column
    :return compiled model, which represents the one with the highest accuracy_score
    '''
    svm_model = svm.SVC().fit(df['train'], y['train'])
    neighbors = KNeighborsClassifier().fit(df['train'], y['train'])
    tree = DecisionTreeClassifier().fit(df['train'], y['train'])
    sgd = SGDClassifier().fit(df['train'], y['train'])
    grad = GradientBoostingClassifier().fit(df['train'], y['train'])
    adaboost = AdaBoostClassifier().fit(df['train'], y['train'])
    forest = RandomForestClassifier().fit(df['train'], y['train'])
    network = MLPClassifier().fit(df['train'], y['train'])

    list_models = [svm_model, neighbors, tree, sgd, grad, adaboost, forest, network]
    scores = []

    for model in list_models:
        scores.append(accuracy_score(model.predict(df['test']), y['test']))

    return list_models[scores.index(max(scores))]


def a_svm(df_1, y, json_file, trained=True):
    '''
    svm training function
    :param df_1: dataframe
    :param y: is the target column
    :trained: whether the model is already trained or not
    :return compiled svm model
    '''

    degree = 3
    gamma = 'scale'
    kernel = 'rbf'

    if 'params' in json_file:
        parameters = json_file['params']
        if 'degree' in json_file['degree']:
            degree = parameters['degree']
        if 'gamma' in json_file['params']:
            gamma = parameters['gamma']
        if 'kernel' in json_file['params']:
            kernel = parameters['kernel']

    clf = svm.SVC(degree=degree, gamma=gamma, kernel=kernel)

    if trained:
        clf.fit(df_1, y)

    return clf


def nearest_neighbors(df_1, y, json_file, trained=True):
    '''
    nearest_neighbors training function
    :param df_1: dataframe
    :param y: is the target column
    :trained: whether the model is already trained or not
    :return compiled nearest_neighbors model
    '''

    sample_amount = int(len(df_1) * 0.5)
    sampled_df = df_1.iloc[:sample_amount]
    sampled_y = y.iloc[:sample_amount]

    if 'params' in json_file:
        n_neighbors = 5
        weights = 'uniform'
        algorithm = 'auto'
        alpha = 0.0001

        if 'params' in json_file:
            parameters = json_file['params']
            if 'n_neighbors' in json_file['params']:
                n_neighbors = parameters['n_neighbors']
            if 'weights' in json_file['params']:
                weights = parameters['weights']
            if 'algorithm' in json_file['params']:
                algorithm = parameters['algorithm']
            if 'alpha' in json_file['params']:
                alpha = parameters['alpha']

        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, alpha=alpha)
        clf = neigh.fit(sampled_df, sampled_y)
        return clf
    else:
        min_neighbors = 3
        max_neighbors = 13
        best_score = 0
        best_model = None
        best_neighbors = 0

        if trained:
            for x in range(min_neighbors, max_neighbors):
                neigh = KNeighborsClassifier(n_neighbors=x)
                neigh.fit(sampled_df, sampled_y)
                a_score = accuracy_score(neigh.predict(sampled_df), sampled_y)
                if a_score > best_score:
                    best_score = a_score
                    best_neighbors = x
                    best_model = neigh

            model = KNeighborsClassifier(n_neighbors=best_neighbors).fit(df_1, y)

        return model


def a_tree(df_1, y, json_file, trained=True):
    '''
    decision tree training function
    :param df_1: dataframe
    :param y: is the target column
    :trained: whether the model is already trained or not
    :return compiled decision tree model
    '''
    criterion = 'gini'
    splitter = 'best'
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 1

    if 'params' in json_file:
        parameters = json_file['params']
        if 'criterion' in json_file['params']:
            criterion = parameters['criterion']
        if 'splitter' in json_file['params']:
            splitter = parameters['splitter']
        if 'max_depth' in json_file['params']:
            max_depth = parameters['max_depth']
        if 'min_samples_split' in json_file['params']:
            min_samples_split = parameters['min_samples_split']
        if 'min_samples_leaf' in json_file['params']:
            min_samples_leaf = parameters['min_samples_leaf']

    tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    if trained:
        tree.fit(df_1, y)

    return tree


def sgd(df_1, y, json_file, trained=True):
    '''
    sgd  training function
    :param df_1: dataframe
    :param y: is the target column
    :trained: whether the model is already trained or not
    :return compiled sgd model
    '''

    loss = 'hinge'
    alpha = 0.0001
    fit_intercept = True
    max_iter = 1000

    if 'params' in json_file:
        parameters = json_file['params']
        if 'loss' in json_file['params']:
            loss = parameters['loss']
        if 'alpha' in json_file['params']:
            alpha = parameters['alpha']
        if 'fit_intercept' in json_file['params']:
            fit_intercept = parameters['fit_intercept']
        if 'max_iter' in json_file['params']:
            max_iter = parameters['max_iter']

    clf = SGDClassifier(loss=loss, alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)

    if trained:
        clf.fit(df_1, y)

    return clf


def gradient_boosting(df_1, y, json_file, trained=True):
    '''
    gradient_boosting  training function
    :param df_1: dataframe
    :param y: is the target column
    :trained: whether the model is already trained or not
    :return compiled gradient_boosting model
    '''
    loss = 'deviance'
    learning_rate = 0.1
    n_estimators = 100
    criterion = 'friedman_mse'

    if 'params' in json_file:
        parameters = json_file['params']
        if 'loss' in json_file['params']:
            loss = parameters['loss']
        if 'learning_rate' in json_file['params']:
            learning_rate = parameters['learning_rate']
        if 'n_estimators' in json_file['params']:
            n_estimators = parameters['n_estimators']
        if 'criterion' in json_file['params']:
            criterion = parameters['criterion']

    clf = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                     criterion='friedman_mse')

    if trained:
        clf.fit(df_1, y)

    return clf


def adaboost(df_1, y, json_file, trained=True):
    n_estimators = 50
    learning_rate = 1
    base_estimator = None

    if 'params' in json_file:
        parameters = json_file['params']
        if 'n_estimators' in parameters:
            n_estimators = parameters['n_estimators']
        if 'learning_rate' in parameters:
            learning_rate = parameters['learning_rate']
        if 'base_estimator' in parameters:
            base_estimator = parameters['base_estimator']

    '''
    adaboost  training function
    :param df_1: dataframe
    :param y: is the target column
    :trained: whether the model is already trained or not
    :return compiled adaboost model
    '''
    clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, base_estimator=base_estimator)
    if trained:
        clf.fit(df_1, y)
    return clf


def rf(df_1, y, json_file, trained=True):
    '''
    random forest  training function
    :param df_1: dataframe
    :param y: is the target column
    :trained: whether the model is already trained or not
    :return compiled random forest model
    '''

    n_estimators = 100
    criterion = 'gini'
    max_depth = None
    max_features = 'auto'

    if 'params' in json_file:
        parameters = json_file['params']
        if 'n_estimators' in parameters:
            n_estimators = parameters['n_estimators']
        if 'criterion' in parameters:
            criterion = parameters['criterion']
        if 'max_depth' in parameters:
            max_depth = parameters['max_depth']
        if 'max_features' in parameters:
            max_features = parameters['max_features']

    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                 max_features=max_features)
    if trained:
        clf.fit(df_1, y)
    return clf


def mlp(df_1, y, json_file, trained=True):
    '''
    mlp  training function
    :param df_1: dataframe
    :param y: is the target column
    :trained: whether the model is already trained or not
    :return compiled mlp model
    '''
    activation = 'relu'
    solver = 'adam'
    alpha = 0.0001
    batch_size = 'auto'
    learning_rate = 'constant'

    if 'params' in json_file:
        parameters = json_file['params']
        if 'activation' in parameters:
            activation = parameters['activation']
        if 'solver' in parameters:
            solver = parameters['solver']
        if 'alpha' in parameters:
            alpha = parameters['alpha']
        if 'batch_size' in parameters:
            batch_size = parameters['batch_size']
        if 'learning_rate' in parameters:
            learning_rate = parameters['learning_rate']

    clf = MLPClassifier(activation=activation, solver=solver, alpha=alpha, batch_size=batch_size,
                        learning_rate=learning_rate)
    if trained:
        clf.fit(df_1, y)
    return clf


def svm_stroke(df_1, y):
    '''
    group of svms training function
    :param df_1: dataframe
    :param y: is the target column
    :return compiled svm model
    '''
    models = []
    scores = []

    clf_1 = svm.SVC().fit(df_1, y)
    clf_2 = svm.SVC(kernel='rbf').fit(df_1, y)
    clf_3 = svm.SVC(kernel='poly').fit(df_1, y)

    models.append(clf_1)
    models.append(clf_2)
    models.append(clf_3)

    for model in models:
        scores.append(accuracy_score(model.predict(df_1), y))

    return models[scores.index(max(scores))]


def ensemble_stroke(df_1, y):
    '''
    group of ensembles training function
    :param df_1: dataframe
    :param y: is the target column
    :return compiled ensemble model
    '''
    models = []
    scores = []

    grad = GradientBoostingClassifier().fit(df_1, y)
    adaboost = AdaBoostClassifier().fit(df_1, y)
    forest = RandomForestClassifier().fit(df_1, y)
    extra = ExtraTreesClassifier().fit(df_1, y)

    models.append(grad)
    models.append(adaboost)
    models.append(forest)
    models.append(extra)

    for model in models:
        scores.append(accuracy_score(model.predict(df_1), y))

    return models[scores.index(max(scores))]
