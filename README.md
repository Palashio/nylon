# papyrus
 Grammerized ML framework.
 
[![Build Status](https://www.travis-ci.com/Palashio/papyrus.svg?token=MFVyVfFQAs3abW7hagzw&branch=main)](https://www.travis-ci.com/Palashio/papyrus)
[![Downloads](https://pepy.tech/badge/papyrus-ai)](https://pepy.tech/project/papyrus-ai)
[![Package](https://img.shields.io/pypi/v/papyrus-ai)


Papyrus offers a high-level natural language representation of machine learning. It allows you to interact with the complex components of the ML pipeline with the english language. 


## Installation

Install latest release version:

```
pip install -U libra
```

Install directory from github:

```
git clone https://github.com/Palashio/libra.git
cd libra
pip install .
```

## Usage: the basics

Papyrus works through the ```papyrusProcessor``` object. When initializing an object, a dataset in the form of a .csv or .xs file should be passed to it by path:

```python
papyrus_object = papyrusProcessor('dataset.csv')
```
