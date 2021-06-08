from papyrus.query import papyrusProcessor

import unittest


def make_orderer():
    order = {}

    def ordered(f):
        order[f.__name__] = len(order)
        return f

    def compare(a, b):
        return [1, -1][order[a] < order[b]]

    return ordered, compare


ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


class TestQueries(unittest.TestCase):
    basic_process = papyrusProcessor('./housing.csv')
    without_preprocessor = papyrusProcessor('./housing.csv')
    multiple_models = papyrusProcessor('./housing.csv')
    all_analysis = papyrusProcessor('./housing.csv')
    """
    TEST QUERIES

    Tests some queries in queries.py
    """

    @ordered
    def test_basic_json(self):
        value = self.basic_process.run('./data_storage/json/basic.json')
        self.assertTrue(str(type(value)) == str(type(self.basic_process)))

    @ordered
    def test_without_preprocessor_json(self):
        value = self.without_preprocessor.run('./data_storage/json/without_preprocessor.json')
        self.assertTrue(str(type(value)) == str(type(self.without_preprocessor)))

    @ordered
    def test_multiple_models(self):
        value = self.multiple_models.run('./data_storage/json/multiple_models.json')
        self.assertTrue(str(type(value)) == str(type(self.multiple_models)))

    @ordered
    def test_multiple_models(self):
        value = self.all_analysis.run('./data_storage/json/all_analysis_methods.json')
        self.assertTrue(str(type(value)) == str(type(self.all_analysis)))

if __name__ == '__main__':
    unittest.main()