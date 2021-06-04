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

def default_modeling(df, y):
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

def a_svm(df_1, y, trained=True):
    clf = svm.SVC()
    if trained:
        clf.fit(df_1, y)

    return clf

def nearest_neighbors(df_1, y, trained=True):
    neigh = KNeighborsClassifier()

    min_neighbors = 3
    max_neighbors = 13
    best_score = 0
    best_model = None
    best_neighbors = 0

    sample_amount = int(len(df_1) * 0.5)
    sampled_df = df_1.iloc[:sample_amount]
    sampled_y = y.iloc[:sample_amount]


    if trained:
        for x in range(min_neighbors, max_neighbors):
            neigh = KNeighborsClassifier(n_neighbors=x)
            neigh.fit(sampled_df, sampled_y)
            a_score = accuracy_score(neigh.predict(sampled_df), sampled_y)
            if a_score > best_score:
                best_score=a_score
                best_neighbors=x
                best_model=neigh

        model = KNeighborsClassifier(n_neighbors=best_neighbors).fit(df_1, y)
        
    return model


def a_tree(df_1, y, trained=True):
    tree = DecisionTreeClassifier()
    if trained:
        tree.fit(df_1, y)

    return tree


def sgd(df_1, y, trained=True):
    clf = SGDClassifier()
    if trained:
        clf.fit(df_1, y)

    return clf


def gradient_boosting(df_1, y, trained=True):
    clf = GradientBoostingClassifier()
    if trained:
        clf.fit(df_1, y)

    return clf


def adaboost(df_1, y, trained=True):
    clf = AdaBoostClassifier()
    if trained:
        clf.fit(df_1, y)
    return clf


def rf(df_1, y, trained=True):
    clf = RandomForestClassifier()
    if trained:
        clf.fit(df_1, y)
    return clf

def mlp(df_1, y, trained=True):
    clf = MLPClassifier()
    if trained:
        clf.fit(df_1, y)
    return clf

def svm_stroke(df_1 , y):
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



