from nylon.supplementaries.main import dataset_initializer
from nylon.supplementaries.handlers import (preprocess_module, modeling_module, analysis_module)
import uuid


class Polymer:
    '''
    Constructor for the Polymer class.
    :param dataset_path is the path to the dataset
    :param save_model: whether the model should be saved in a .sav file
    '''

    def __init__(self, dataset_path, save_model=False, custom_files=[]):
        self.df = dataset_path
        self.json_file = None
        self.y = None
        self.model = None
        self.results = None
        self.custom_files = custom_files
        self.history = {}
        self.dataframe = None
        self.latest_id = None

    def run(self, json_file_path, as_dict=False):
        '''
        Runs the dataset on a json file specification
        :param json_file_path path to json file for
        '''

        if self.model is not None:
            self.update_history(json_file_path)

        request_info = {'df': self.df, 'json': json_file_path, 'y': None, 'model': 'None', 'analysis': None,
                        'custom': self.custom_files}

        pipeline = [
            dataset_initializer,
            preprocess_module,
            modeling_module,
            analysis_module
        ]

        for a_step in pipeline:
            request_info = a_step(request_info)

        self.set_class_after_run(request_info)

        return self

    def update_history(self, json_file_path):
        self.history[self.latest_id] = {'df': self.df, 'json': json_file_path, 'y': self.y,
                                        'model': self.model, 'results': self.results}


    def set_class_after_run(self, request_info):
        new_id = str(uuid.uuid4())
        self.latest_id = new_id

        self.results = request_info['analysis']
        self.model = request_info['model']
        self.json_file = request_info['json']
        self.y = request_info['y']
        self.df = request_info['df']