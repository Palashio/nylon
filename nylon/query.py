from nylon.supplementaries.main import dataset_initializer
from nylon.supplementaries.handlers import (preprocess_module, modeling_module, analysis_module)
import uuid
class nylonProcessor:
    '''
    Constructor for the nylonProcessor class.
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
        self.id = uuid.uuid4()
    def run(self, json_file_path):
        '''
        Runs the dataset on a json file specification
        :param json_file_path path to json file for
        '''

        if self.model is not None:
            self.history[str(self.id)] = {'df': self.df, 'json': json_file_path, 'y': self.y, 'model': 'self.model', 'results': self.results}


        request_info = {'df': self.df, 'json': json_file_path, 'y': None, 'model': 'None', 'analysis': None, 'custom': self.custom_files}
        pipeline = [
            dataset_initializer,
            preprocess_module,
            modeling_module,
            analysis_module
        ]

        for a_step in pipeline:
            request_info = a_step(request_info)

        self.results = request_info['analysis']
        self.model = request_info['model']
        self.df = request_info['df']

        return self
