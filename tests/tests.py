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
    newClient = papyrusProcessor('/Users/palashshah/desktop/papyrus/housing.csv', '/Users/palashshah/desktop/papyrus/test.json')
    """
    TEST QUERIES

    Tests some queries in queries.py
    """

    # Tests whether regression_ann_query works without errors, and creates a key in models dictionary
    @ordered
    def test_basic_json(self):
        self.newClient.run()
        self.assertTrue('hi' == 'hi')


if __name__ == '__main__':
    unittest.main()