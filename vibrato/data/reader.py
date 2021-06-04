import pandas as pd

class DataReader():
    def __init__(self, in_mem_data, dataset_name):
        self.data = in_mem_data
        self.dataset_name = dataset_name

    def data_reader(self):
        frame = 0
        if 'type' not in self.data:
            data_type = self.type_detector(self.dataset_name)
        else:
            data_type = self.data['type']

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
        if data[-3:] == 'csv':
            return 'csv'
        if data[-2:] == 'xs':
            return 'xs'
        if data[-4:] == 'json':
            return 'json'

    def csv(self, dataset_name, drop=[]):

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
        try:
            df = pd.read_excel(datset_name)
        except:
            raise Exception("The file path you provided -- {} -- does not exist".format(datset_name))
        for element in drop:
            try:
                del df[element]
            except:
                raise Exception("Your column -- {} --  is not in the dataframe".format(element))