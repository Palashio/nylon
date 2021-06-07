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
    basic_process = papyrusProcessor('./data/json/basic.json', './housing.csv')
    without_preprocessor = papyrusProcessor('./data/json/without_preprocessor.json', './housing.csv')
    multiple_models = papyrusProcessor('./data/json/multiple_models.json', './housing.csv')
    all_analysis = papyrusProcessor('./data/json/all_analysis_methods.json', './housing.csv')
    """
    TEST QUERIES

    Tests some queries in queries.py
    """

    @ordered
    def test_basic_json(self):
        random_dict = {}
        value = self.basic_process.run()
        self.assertTrue(str(type(value)) == str(type(random_dict)))

    @ordered
    def test_without_preprocessor_json(self):
        random_dict = {}
        value = self.without_preprocessor.run()
        self.assertTrue(str(type(value)) == str(type(random_dict)))

    @ordered
    def test_multiple_models(self):
        random_dict = {}
        value = self.multiple_models.run()
        self.assertTrue(str(type(value)) == str(type(random_dict)))

    @ordered
    def test_multiple_models(self):
        random_dict = {}
        value = self.all_analysis.run()
        self.assertTrue(str(type(value)) == str(type(random_dict)))



if __name__ == '__main__':
    unittest.main()