
from pandas.api.types import is_numeric_dtype

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from nylon.preprocessing.preprocessing import process_dates
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer, LabelEncoder)

def handle_scaling(preprocess, df):
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
    return df

def handle_min_max(preprocess, df):
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
    return df

def handle_label_encode(preprocess, df):
    columns = preprocess['label-encode']

    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        enc = LabelEncoder()
        resulting_encoder = enc.fit_transform(df[column])
        df = df.assign(ocean_proximity=resulting_encoder)
    return df

def handle_ordinal(preprocess, df):
    columns = preprocess['ordinal']

    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        enc = OrdinalEncoder()
        df[column] = enc.fit_transform(np.array(df[column]).reshape(-1, 1))

    return df


def handle_filling(preprocess, df):
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

    return df
def handle_importance(preprocess, df, json_file):
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
        top_indices = np.argpartition(importance, number * -1)[number * 1:]

        for index, column in enumerate(df.columns):
            if index not in top_indices:
                del df[column]

    df[target] = y.values

    return df

def handle_one_hot(preprocess, df, json_file):
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
    return df

def handle_text(preprocess, df):
    columns = preprocess['clean-text']
    if isinstance(columns, str):
        columns = [columns]

    df = text_preprocessing(df, columns)

    return df

def handle_embedding(preprocess, df):
    columns = preprocess['embed']
    if isinstance(columns, str):
        columns = [columns]

    df = embedding_preprocessor(df, columns)

    return df

def handle_dates(preprocess, df):
    columns = preprocess['dates']

    if isinstance(columns, str):
        columns = [columns]

    df = process_dates(df)



    return df


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

