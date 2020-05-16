from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from ModelSearcher import ModelSearcherABC

import numpy as np


class LogisticRegressionSearcher(ModelSearcherABC):
    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l2"]}

    def __init__(self, X_train, y_train, seed, cv):
        self.X_train = X_train
        self.y_train = y_train
        self.seed = seed
        self.cv = cv

    def compute_grid_search(self):
        logreg = LogisticRegression()
        logreg_cv = GridSearchCV(logreg, self.grid, cv=self.cv)
        logreg_cv.fit(self.X_train, self.y_train)
        best_score = logreg_cv.best_score_
        logreg_best_params = logreg_cv.best_params_

        final_model = LogisticRegression(random_state=self.seed,  **logreg_best_params)
        final_model.fit(self.X_train, self.y_train)

        return final_model, best_score

