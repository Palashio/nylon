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
    newClient = papyrusProcessor('/Users/palashshah/desktop/papyrus/data/json/basic.json', '/Users/palashshah/desktop/papyrus/housing.csv')
    """
    TEST QUERIES

    Tests some queries in queries.py
    """

    # Tests whether regression_ann_query works without errors, and creates a key in models dictionary
    @ordered
    def test_basic_json(self):
        value = self.newClient.run()
        random_dict = {}
        self.assertTrue(str(type(value)) == str(type(random_dict)))


if __name__ == '__main__':
    unittest.main()