# papyrus
 Grammerized ML framework.
 
[![Build Status](https://www.travis-ci.com/Palashio/papyrus.svg?token=MFVyVfFQAs3abW7hagzw&branch=main)](https://www.travis-ci.com/Palashio/papyrus)
[![Downloads](https://pepy.tech/badge/papyrus-ai)](https://pepy.tech/project/papyrus-ai)
[![Package](https://img.shields.io/pypi/v/papyrus-ai)


Papyrus offers a high-level natural language representation of machine learning. It allows you to interact with the complex components of the ML pipeline with the english language. 


## Installation

Install latest release version:

```
pip install -U papyrus-ai
```

Install directory from github:

```
git clone https://github.com/Palashio/papyrus.git
cd libra
pip install .
```

## Usage: the basics

Papyrus works through the ```papyrusProcessor``` object. When initializing an object, a dataset in the form of a .csv or .xs file should be passed to it by path:

```python
papyrus_object = papyrusProcessor('housing.csv')
```

Now, it's time to create a specifications file using the papyrus grammar. Here's a basic one, that lets papyrus handle most of the work. 

```json
{
    "data": {
        "target": "ocean_proximity"
    },
    "preprocessor": {
        "fill": "ALL",
        "label-encode": "ocean_proximity"
    }
}
```

Now, we can override more components to take advantage of the built in ensembling of SVM's, and nearest neighbors modeling in papyrus. 
```python
 json_file = {
    "data": {
        "target": "ocean_proximity"
    },
    "preprocessor": {
        "fill": "ALL",
        "label-encode": "ocean_proximity"
    },
    "modeling": {
        "type": ["svms", "neighbors"]
    }
}
```

Now we can call,

```python
papyrus_object.run(json_file)
```

More docs can be found at [here](docs.paraglide.ai)!
