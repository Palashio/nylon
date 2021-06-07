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
    basic_process = papyrusProcessor('/Users/palashshah/desktop/papyrus/data/json/basic.json', '/Users/palashshah/desktop/papyrus/housing.csv')
    """
    TEST QUERIES

    Tests some queries in queries.py
    """

    @ordered
    def test_basic_json(self):
        value = self.basic_process.run()
        random_dict = {}
        self.assertTrue(str(type(value)) == str(type(random_dict)))



if __name__ == '__main__':
    unittest.main()