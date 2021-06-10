from nylon.query import nylonProcessor

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
    basic_process = nylonProcessor('./housing.csv')
    without_preprocessor = nylonProcessor('./housing.csv')
    multiple_models = nylonProcessor('./housing.csv')
    all_analysis = nylonProcessor('./housing.csv')
    with_trim = nylonProcessor('./housing.csv')
    data_custom = nylonProcessor('./housing.csv')
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

    @ordered
    def test_trim(self):
        value = self.with_trim.run('./data_storage/json/with_trim.json')
        self.assertTrue(str(type(value)) == str(type(self.with_trim)))

    @ordered
    def test_data_custom(self):
        value = self.data_custom.run('./data_storage/json/data_custom.json')
        self.assertTrue(str(type(value)) == str(type(self.data_custom)))

if __name__ == '__main__':
    unittest.main()