import json

def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def workflows(name):
    if name == 'basic':
        return read_json('./data_storage/standard_workflows/basic.json')

    if name == 'ensembles':
        return read_json('./data_storage/standard_workflows/full_preprocessor.json')
