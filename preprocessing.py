import pathlib
import win32com.client

import pandas as pd
import sklearn.model_selection

import nltk


def import_excel(file: pathlib.Path, password: str, start_cell: tuple[int, int],
                           end_cell: tuple[int, int], sheet: int = 1):
    xlApp = win32com.client.Dispatch("Excel.Application")

    # https://docs.microsoft.com/en-ca/office/vba/api/Excel.Workbooks.Open
    xlwb = xlApp.Workbooks.Open(file, False, True, None, password)

    # https://docs.microsoft.com/en-ca/office/vba/api/excel.worksheet
    xlws = xlwb.Sheets(sheet)
    return pd.DataFrame(list(xlws.Range(xlws.Cells(*start_cell), xlws.Cells(*end_cell)).Value))


def preprocess(df: pd.DataFrame):
    df = df.applymap(nltk.tokenize.word_tokenize)

    lemmatizer = nltk.stem.WordNetLemmatizer()
    df = df.applymap(lambda w: [lemmatizer.lemmatize(t) for t in w])

    return df


def make_datasets_from_df(data: pd.DataFrame, coding: pd.DataFrame):
    X, y = data.stack(), coding.stack()

    # TODO: add stratify parameter after data is labelled
    return sklearn.model_selection.train_test_split(X, y, test_size=0.3)


if __name__ == '__main__':
    import yaml

    with open('settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)['SETTINGS']

    data_dir = pathlib.Path().resolve()/'.salg'
    file_name = data_dir/'SALG-Instrument-8601.xlsx'

    # free response - r[14], r[15], r[27], r[35], r[40], r[44], r[45], r[54], r[55], r[70], r[76], r[80], r[92], r[93], r[94]
    cols = [14, 15, 27, 35, 40, 44, 45, 54, 55, 70, 76, 80, 92, 93, 94]

    data = import_excel(file_name, settings['PASSWORD'], (2, 3), (351, 97), 3).iloc[:, cols]
    data = preprocess(data)

    coding = import_excel(file_name, settings['PASSWORD'], (2, 2), (351, 16), 4)

    X_train, X_test, y_train, y_test = make_datasets_from_df(data, coding)
