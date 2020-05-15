from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from ModelSearcher import ModelSearcherABC


class RandomForestModelSearcher(ModelSearcherABC):
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }

    def __init__(self, X_train, y_train, seed, cv):
        self.X_train = X_train
        self.y_train = y_train
        self.seed = seed
        self.cv = cv

    def compute_grid_search(self):
        rfc = RandomForestClassifier(random_state=42)
        CV_rfc = GridSearchCV(estimator=rfc, param_grid=self.param_grid, cv=self.cv)
        CV_rfc.fit(self.X_train, self.y_train)
        best_params_rf = CV_rfc.best_params_
        best_score = CV_rfc.best_score_

        final_model = RandomForestClassifier(random_state=self.seed, **best_params_rf)
        final_model.fit(self.X_train, self.y_train)

        return final_model, best_score
