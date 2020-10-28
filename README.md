# HyperclassifierSearch

## General info
HyperclassifierSearch allows to train multiple classifiers/pipelines in Python with GridSearchCV or RandomizedSearchCV.

## Installation
`pip install HyperclassifierSearch`

## Requirements
The code was developed in Python 3. The execution needs Pandas and scikit-learn, i.e. GridSearchCV and RandomizedSearchCV.

## Enhancements and credits
The package is build based on code from [David Batista](https://github.com/davidsbatista/machine-learning-notebooks/blob/master/hyperparameter-across-models.ipynb).

1. documentation enhancements:
- examples how to search the best model over multiple Pipelines using different classifiers
- added code documentation including docstrings

2. functionality enhancements:
- added option to use RandomizedSearchCV
- the best overall model is provided by train_model()
- output dataframe is simplified as standard option

## Examples
Please refer to `HyperclassifierSearch examples.ipynb` in the root folder.
