from RandomForest import RandomForestModelSearcher
from LogReg import LogisticRegressionSearcher

import pickle
import os
import concurrent.futures


class AutoMlBinary:
    def __init__(self, X_train, y_train, seed, cv, models_names, n_workers):
        self.X_train = X_train
        self.y_train = y_train
        self.seed = seed
        self.cv = cv
        self.n_workers = n_workers
        self.searchers = self._get_models(models_names)
        self.best_model = None

    def fit(self):
        models = list()
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = [executor.submit(searcher.compute_grid_search) for searcher in self.searchers]
            for f in concurrent.futures.as_completed(results):
                models.append(f.result())

        self._set_best_model(models)

    def score(self, X_test, y_test):
        score = self.best_model.score(X_test, y_test)
        return score

    def predict(self, X_data):
        prediction = self.best_model.predict(X_data)
        return prediction

    def save(self, path):
        path_with_name = os.path.join(path, 'my_dumped_classifier.pkl')

        with open(path_with_name, 'wb') as fid:
            pickle.dump(self.best_model, fid)

    def _get_models(self, models_names):#TODO Factory pattern?
        searchers = list()

        if 'Random_forest' in models_names:
            searchers.append(RandomForestModelSearcher(self.X_train, self.y_train, self.seed, self.cv))

        if 'Logistic_regression' in models_names:
            searchers.append(LogisticRegressionSearcher(self.X_train, self.y_train, self.seed, self.cv))

        return searchers

    def _set_best_model(self, models):
        models.sort(key=lambda x: x[1], reverse=True)
        self.best_model = models[0][0]
