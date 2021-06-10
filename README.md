<div align="center">
 
<img src="nylonlogo.jpg" alt="drawing" width="500"/>
 
 
 
[![Build Status](https://www.travis-ci.com/Palashio/nylon.svg?token=MFVyVfFQAs3abW7hagzw&branch=main)](https://www.travis-ci.com/Palashio/nylon)
[![Downloads](https://pepy.tech/badge/papyrus-ai)](https://pepy.tech/project/nylon-ai)
[![Package](https://img.shields.io/pypi/v/nylon-ai)

nylon offers a high-level natural language representation of machine learning. It allows you to interact with the complex components of the ML pipeline with the english language.

 </div>
## Installation

Install latest release version:

```
pip install -U nylon-ai
```

Install directory from github:

```
git clone https://github.com/Palashio/papyrus.git
cd papyrus-ai
pip install .
```

## Usage: the basics

nylon works through the `nylonProcessor` object. When initializing an object, a dataset in the form of a .csv or .xs file should be passed to it by path:

```python
nylon_object = nylonProcessor('housing.csv')
```

Now, it's time to create a specifications file using the nylon grammar. Here's a basic one, that lets nylon handle most of the work.

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

Now, we can override more components to take advantage of the built in ensembling of SVM's, and nearest neighbors modeling in nylon.

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
nylon_object.run(json_file)
```

More docs can be found at [here](docs.paraglide.ai)!

## Contact

Shoot me an email at [hello@paraglide.ai](mailto:hello@paraglide.ai) if you'd like to get in touch!

Follow me on [twitter](https://twitter.com/_pshah) for updates and my insights about modern AI!
