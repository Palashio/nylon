from nylon.query import Polymer

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
    basic_process = Polymer('./data_storage/datasets/housing.csv')
    without_preprocessor = Polymer('./data_storage/datasets/housing.csv')
    multiple_models = Polymer('./data_storage/datasets/housing.csv')
    all_analysis = Polymer('./data_storage/datasets/housing.csv')
    with_trim = Polymer('./data_storage/datasets/housing.csv')
    data_custom = Polymer('./data_storage/datasets/housing.csv')
    voting_classifier = Polymer('./data_storage/datasets/housing.csv')
    test_history = Polymer('./data_storage/datasets/housing.csv')
    using_strokes = Polymer('./data_storage/datasets/housing.csv')
    test_tree = Polymer('./data_storage/datasets/housing.csv')

    """
    TEST QUERIES

    Tests some queries in queries.py
    """

    @ordered
    def test_basic_json(self):
        '''
        Tests a simple JSON file located in basic.json
        '''
        value = self.basic_process.run('./data_storage/json/basic.json')
        self.assertTrue(str(type(value)) == str(type(self.basic_process)))

    @ordered
    def test_without_preprocessor_json(self):
        '''
        Tests a JSON file without the preprocessing section specified.
        '''
        value = self.without_preprocessor.run('./data_storage/json/without_preprocessor.json')
        self.assertTrue(str(type(value)) == str(type(self.without_preprocessor)))

    @ordered
    def test_multiple_models(self):
        '''
        Tests a JSON file with multiple models specified in the modeling section.
        '''
        value = self.multiple_models.run('./data_storage/json/multiple_models.json')
        self.assertTrue(str(type(value)) == str(type(self.multiple_models)))

    @ordered
    def test_all_analysis(self):
        '''
        Tests a JSON file without analysis specified in order to test all of them.
        '''
        value = self.all_analysis.run('./data_storage/json/all_analysis_methods.json')
        self.assertTrue(str(type(value)) == str(type(self.all_analysis)))

    @ordered
    def test_trim(self):
        '''
        Tests a JSON file with the trimming function in the data section activated.
        '''
        value = self.with_trim.run('./data_storage/json/with_trim.json')
        self.assertTrue(str(type(value)) == str(type(self.with_trim)))

    @ordered
    def test_voting(self):
        '''
        Tests a JSON file with the modeling section including a voting classifier.
        '''
        value = self.voting_classifier.run('./data_storage/json/voting_classifiers.json')
        self.assertTrue(str(type(value)) == str(type(self.voting_classifier)))

    @ordered
    def test_voting(self):
        '''
        Tests a JSON file with the modeling section including a voting classifier.
        '''
        value = self.voting_classifier.run('./data_storage/json/voting_classifiers.json')
        self.assertTrue(str(type(value)) == str(type(self.voting_classifier)))

    @ordered
    def test_history_functionality(self):
        '''
        Uses the basic.json specifications in order to test the storage of the information after multiple runs in the .history section.
        '''
        self.test_history.run('./data_storage/json/basic.json')
        self.test_history.run('./data_storage/json/basic.json')
        print(self.test_history.history)
        self.assertTrue(len(self.test_history.history) == 1)

    @ordered
    def test_strokes(self):
        '''
        Tests a JSON file with the modeling section including a voting classifier.
        '''
        value = self.using_strokes.run('./data_storage/json/using_strokes.json')
        self.assertTrue(str(type(value)) == str(type(self.using_strokes)))

    @ordered
    def test_fetcher(self):
        from nylon.fetcher import workflows

        basic_loaded = workflows('basic')
        ensemble_loaded = workflows('ensembles')

        self.assertTrue(isinstance(basic_loaded, dict) and isinstance(basic_loaded, dict))

    @ordered
    def testing_tree(self):
        value = self.test_tree.run('./data_storage/json/basic.json')
        self.assertTrue(str(type(value)) == str(type(self.test_tree)))


if __name__ == '__main__':
    unittest.main()
