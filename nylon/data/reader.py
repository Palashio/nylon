import pandas as pd

import warnings

warnings.filterwarnings('ignore')


class DataReader():
    '''
    Constructor for the DataReader class.
    Data Reader class takes in the information given by the user, and appropriately reads the data in. Detects for filetype, and other information passed to it
    :param in_mem_data: data file as a path
    :param dataset_name: name of the dataset
    '''

    def __init__(self, in_mem_data, dataset_name):
        self.data = in_mem_data
        self.dataset_name = dataset_name

    def elements_to_drop(self):
        dropping = []
        if "drop" not in self.data:
            pass
        else:
            for element in self.data["drop"]:
                if element == self.data["target"]:
                    raise Exception(
                        "Your target of --  {} -- cannot be the same as the columns you're trying to drop.".format(
                            self.data["target"]))
                dropping.append(element)
        return dropping

    def data_reader(self):
        '''
        function to activate the data reading
        :return dataframe that's read into the system as a pandas dataframe object
        '''
        frame = 0
        if 'type' not in self.data:
            data_type = self.type_detector(self.dataset_name)
        else:
            data_type = self.data['type']

        dropping = self.elements_to_drop()

        if data_type == 'csv':
            frame = self.csv(self.dataset_name, drop=dropping)
            if self.data['data']['target'] not in frame.columns:
                raise Exception("The target you provided -- {} -- is not in the dataframe.".format(self.data["target"]))
        if data_type == 'xs':
            frame = self.xs(self.data, drop=dropping)
            if self.data['target'] not in frame.columns:
                raise Exception("The target you provided -- {} -- is not in the dataframe.".format(self.data["target"]))
        if 'trim' not in self.data:
            pass
        else:
            if self.data['trim'] < 0.1 or self.data['train'] > 0.9:
                raise Exception("Trimming amounts should be between the value 0.1 and 0.9.")
            target_size = int(self.data['trim'] * len(frame))

            while target_size > len(frame):
                frame = frame.iloc[1:]

        return frame

    def type_detector(self, data):
        '''
        helper function to take the file name and figure out what sort of file it is
        :param data the string name of the file
        :return file extension name
        '''
        if data[-3:] == 'csv':
            return 'csv'
        if data[-2:] == 'xs':
            return 'xs'
        if data[-4:] == 'json':
            return 'json'

    def csv(self, dataset_name, drop=[]):
        '''
        function to read in csv file
        :param dataset_name is the path to the actual dataset
        :param drop list of columns to drop, not activated yet
        :return pandas dataframe
        '''
        try:
            df = pd.read_csv(dataset_name)
        except:
            raise Exception("The file path you provided -- {} -- does not exist".format(str(self.data)))
        for element in drop:
            try:
                del df[element]
            except:
                raise Exception("Your column -- {} --  is not in the dataframe".format(element))
        return df

    def xs(self, datset_name, drop=[]):
        '''
        function to read in excel file
        :param dataset_name is the path to the actual dataset
        :param drop list of columns to drop, not activated yet
        :return pandas dataframe
        '''
        try:
            df = pd.read_excel(datset_name)
        except:
            raise Exception("The file path you provided -- {} -- does not exist".format(datset_name))
        for element in drop:
            try:
                del df[element]
            except:
                raise Exception("Your column -- {} --  is not in the dataframe".format(element))
