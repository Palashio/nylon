from pandas.core.algorithms import mode
from nylon.supplementaries.main import dataset_initializer
from nylon.supplementaries.handlers import (preprocess_module, modeling_module, analysis_module, perform_inference)
import uuid
import pandas as pd

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
        self.debug = False
        self.col_names = None
        self.pca_model = None
        self.transforms = None

    def run(self, json_file_path, as_dict=False, perform_PCA = True):
        '''
        Runs the dataset on a json file specification
        :param json_file_path path to json file for
        '''
        new_id = str(uuid.uuid4())
        self.latest_id = new_id
        
        if self.model is not None:
            self.update_history(json_file_path)

        request_info = {'df': self.df, 'json': json_file_path, 'y': None, 'model': 'None', 'analysis': None,
                        'custom': self.custom_files, 'pca': perform_PCA}

        pipeline = None

        if(self.debug):
            pipeline = [dataset_initializer, preprocess_module, modeling_module]
        else: 
            pipeline = [dataset_initializer, preprocess_module, modeling_module, analysis_module]

        for a_step in pipeline:
            request_info = a_step(request_info)

        self.set_class_after_run(request_info)

        return self

    def runwithPCA(self, json_file_path): 

        new_id = str(uuid.uuid4())
        self.latest_id = new_id

        self.runPCA = True
        self.df_trans = None
        self.pca_model = None
        self.column_names = None
        
        if self.model is not None:
            self.update_history(json_file_path)

        request_info = {'df': self.df_path, 'json': json_file_path, 'y': None, 'model': 'None', 'analysis': None,
                        'custom': self.custom_files, 'pca': self.runPCA}
        
        dataset_initializer(request_info)

        preprocess_module(request_info)
        
        pca_dict = request_info['info']
        self.column_names = pca_dict['original_names']
        self.pca_model = pca_dict['pca_model']
        self.df = request_info['df']
        self.df_trans = dict()
        self.df_trans['train'] = self.transformToPrinciple(self.df['train'])
        self.df_trans['test'] = self.transformToPrinciple(self.df['test'])
        request_info['df'] = self.df_trans
        
        modeling_module(request_info)
        analysis_module(request_info)

        self.set_class_after_run(request_info)

        return self

    def update_history(self, json_file_path):
        self.history[self.latest_id] = {'df': self.df, 'json': json_file_path, 'y': self.y,
                                        'model': self.model, 'results': self.results}


    def set_class_after_run(self, request_info):
        self.results = request_info['analysis']
        self.model = request_info['model']
        
        self.json_file = request_info['json']
        self.y = request_info['y']
        self.dataframe = request_info['df']

        self.col_names = request_info['col_names']
        if 'pca_model' in request_info:
            self.pca_model = request_info['pca_model']
        self.transforms = request_info['target-transforms']
    
    def get_results(self, input_file, output_file_path = ""):
        request_info = {'input' : input_file, 'cols' : self.col_names, 'json' : self.json_file, 
        'model'  : self.model, 'result-transforms' : self.transforms}
        if self.pca_model is not None: 
            request_info['pca-model'] = self.pca_model
        output = perform_inference(request_info)
        if(len(output_file_path) > 0):
            if(output_file_path[-3 : ] == 'csv'):
                output.to_csv(output_file_path, index = False)
            elif(output_file_path[-4 : ] == 'json'):
                output.to_json(output_file_path, orient='records')
        return output
    
    def transformToPrinciple(self, values):
        if not self.runPCA:
            return None
        transformed = self.pca_model.transform(values)
        num_columns = transformed.shape[1]
        column_names = ["Principal#" + str(i + 1) for i in range(num_columns)]
        transform_df = pd.DataFrame(transformed, columns = column_names)
        return transform_df
    
    def transformToInput(self, transformed): 
        if not self.runPCA:
            return None
        original = self.pca_model.inverse_transform(transformed)
        num_columns = original.shape[1]
        assert num_columns == len(self.column_names)
        original_df = pd.DataFrame(original, columns = self.column_names)
        return original_df

    def predict(self, file_path):
        the_model = self.model
        df = pd.read_csv(file_path)


        request_info = {'df': df, 'json': self.json_file, 'y': None, 'model': 'None', 'analysis': None,
                        'custom': self.custom_files, 'pca': self.runPCA}

        request_info = preprocess_module_inference(request_info)

        final_df = pd.concat([request_info['df']['train'], request_info['df']['test']], axis=0)

        model = self.model
        output = model.predict(final_df)
        return output