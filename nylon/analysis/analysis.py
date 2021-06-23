from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

def default_analysis(json_file, model, df, y):
    acc_results = acc_score(model, df, y)
    cv_results = cross_score(json_file, df, model, y)
    matrix = confusion(model, df, y)
    precise = precision_calculation(model, df, y)
    recaller = recall_score_helper(model, df, y)
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

def recall_score_helper(model, df, y):
    output = model.predict(df['test'])
    average_value = ('binary' if len(np.unique(y['test'])) == 2 else 'macro')
    return float(recall_score(y['test'], output, average=average_value, labels=np.unique(output)))