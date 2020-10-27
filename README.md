# HyperclassifierTuning

## General info
HyperclassifierTuning allows to train multiple classifiers/pipelines in Python with GridSearchCV or RandomizedSearchCV.

## Installation
The code was developed in Python 3. The execution needs Pandas and GridSearchCV and RandomizedSearchCV from scikit-learn.

(!) ToDo: add code as PyPi package.

Apart from installing the package from PyPi you find the code in the Juypter Notebook 'HyperclassifierSearch.ipynb' including the examples below.

## Example 1: train multiple classifiers
```
# imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# example dataset
from sklearn import datasets
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# define classifiers and parameters for hyperfitting
models = {
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=10000),
    'RandomForestClassifier': RandomForestClassifier()
}
params = {
    'LogisticRegression': { 'C': [0.1, 1, 2] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] }
}

# run search
X_train, X_test, y_train, y_test = train_test_split(X, y)
search = HyperclassifierSearch(models, params)
best_model = search.train_model(X, y, cv=2, iid=False)
search.evaluate_model()
```

## Example 2: add multiple pipelines to example 1
```
# additional imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# define the model including pipelines
models = {
    'LogisticRegression': Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=200))
    ]),
    'RandomForestClassifier': Pipeline([
        ('scale', StandardScaler()),
        ('clf', RandomForestClassifier())
    ])
}
params = {
    'LogisticRegression': { 'clf__C': [0.1, 1, 2] },
    'RandomForestClassifier': { 'clf__n_estimators': [16, 32] }
}

X_train, X_test, y_train, y_test = train_test_split(X, y)
search = HyperclassifierSearch(models, params)
best_model = search.train_model(X, y, cv=10, iid=False)
search.evaluate_model()
```

## Example 3: using RandomizedSearchCV and more exhaustive search compared to example 2
```
# additional imports:
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, make_scorer, accuracy_score, balanced_accuracy_score, fbeta_score

# model and parameter definition in a function
def build_model():
    models = {

        'LogisticRegression': Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(solver='lbfgs'))
        ]),
        'RandomForestClassifier': Pipeline([
            ('scale', StandardScaler()),
            ('clf', RandomForestClassifier())
        ]),
        'AdaBoost': Pipeline([
            ('tfidf', StandardScaler()),
            ('clf', AdaBoostClassifier())  
        ])
    }
    params = {
        'LogisticRegression': { 'clf__C': np.linspace(0.1, 1.0, num=10) },
        'RandomForestClassifier': { 'clf__n_estimators': np.arange(16,32+1) },
        'AdaBoost': { 'clf__n_estimators': np.arange(16,32+1) }
    }
    scorer = make_scorer(fbeta_score, beta=2, average='weighted')
    return models, params, scorer
```

```
models, params, scorer = build_model()
search = HyperclassifierSearch(models, params)
skf = StratifiedKFold(n_splits=5)
best_model = search.train_model(X, y, search='random', scoring=scorer, cv=skf, iid=False)
search.evaluate_model()
```
