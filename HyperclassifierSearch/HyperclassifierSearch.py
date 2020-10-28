# imports
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class HyperclassifierSearch:
    """Train multiple classifiers/pipelines with GridSearchCV or RandomizedSearchCV.

    HyperclassifierTuning implements a "train_model" and "evaluate_model" method.

    "train_model" returns the optimal model according to the scoring metric.

    "evaluate_model" gives the results for all classifiers/pipelines.

    ## Example: ##
    # import of the package (after e.g. 'pip install HyperclassifierSearch')
    from HyperclassifierSearch import HyperclassifierSearch

    # usage dependent imports
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
    """
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.grid_results = {}

    def train_model(self, X_train, y_train, search='grid', **search_kwargs):
        """
        Optimizing over one or multiple classifiers or pipelines.

        Input:
        X : array or dataframe with features; this should be a training dataset
        y : array or dataframe with label(s); this should be a training dataset

        Output:
        returns the optimal model according to the scoring metric

        Parameters:
        search : str, default='grid'
            define the search
            ``grid`` performs GridSearchCV
            ``random`` performs RandomizedSearchCV

        **search_kwargs : kwargs
            additional parameters passed to the search
        """
        grid_results = {}
        best_score = 0

        for key in self.models.keys():
            print('Search for {}'.format(key), '...')
            assert search in ('grid', 'random'), 'search parameter out of range'
            if search=='grid':
                grid = GridSearchCV(self.models[key], self.params[key], **search_kwargs)
            if search=='random':
                grid = RandomizedSearchCV(self.models[key], self.params[key], **search_kwargs)
            grid.fit(X_train, y_train)
            self.grid_results[key] = grid

            if grid.best_score_ > best_score: # return best model
                best_score = grid.best_score_
                best_model = grid

        print('Search is done.')
        return best_model # allows to predict with the best model overall

    def evaluate_model(self, sort_by='mean_test_score', show_timing_info=False):
        """
        Provides sorted model results for multiple classifier or pipeline runs of
        GridSearchCV or RandomizedSearchCV.

        Input: Fitted search object (accesses cv_results_).
        Output: Dataframe with a line for each training run including estimator name, parameters, etc.
        Parameters:
        sort_by: the metric to rank the model results
        """
        results = []
        for key, result in self.grid_results.items():
            print('results round for:', key)
            # get rid of column which is estimator specific, i.e. use df for multiple estimators
            result = pd.DataFrame(result.cv_results_).filter(regex='^(?!.*param_).*')
            if show_timing_info==False: # skip timing info
                result = result.filter(regex='^(?!.*time).*')
            # add column with the name of the estimator
            result = pd.concat((pd.DataFrame({'Estimator': [key] * result.shape[0] }), result), axis=1)
            results.append(result)

        # handle combined classifier results: sort by target metric and remove subset rank scores
        df_results = pd.concat(results).sort_values([sort_by], ascending=False).\
                        reset_index().drop(columns = ['index', 'rank_test_score'])
        return df_results
