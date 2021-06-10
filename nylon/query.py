from nylon.supplementaries.main import dataset_initializer
from nylon.supplementaries.handlers import (preprocess_module, modeling_module, analysis_module)

class nylonProcessor:
    def __init__(self, dataset_path, save_model=False, custom_files = []):
        self.df = dataset_path
        self.json_file = None
        self.y = None
        self.model = None
        self.results = None
        self.custom_files = custom_files
    def run(self, json_file_path):
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