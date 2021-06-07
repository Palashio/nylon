from papyrus.supplementaries.main import dataset_initializer
from papyrus.supplementaries.handlers import (preprocess_module, modeling_module, analysis_module)

class papyrusProcessor:
    def __init__(self, json_file_path, dataset_path, save_model=False):
        self.df, self.json_file = dataset_initializer(dataset_path, json_file_path)
        self.y = None
        self.columns = self.df.columns
        self.model = None

    def run(self):
        request_info = {'df': self.df, 'json': self.json_file, 'y': None, 'model': 'None', 'analysis': None}

        pipeline = [
            preprocess_module,
            modeling_module,
            analysis_module
        ]

        for a_step in pipeline:
            request_info = a_step(request_info)

        return request_info


processor = papyrusProcessor('/Users/palashshah/desktop/papyrus/papyrus/test.json', 'housing.csv')
print(processor.run())