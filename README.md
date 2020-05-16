# AutoMl Library

Блиблиотека на языке python.

## Getting Started

Данная библиотека предназначена для автоматического построения моделей машинного обучения.
В данный момент библиотека ведет перебор моделей и гиперпараметров, для этих моделей. 

### Prerequisites

Для работы необходима сторонняя библиотека sklearn

```
pip install sklearn
```

### Installing

Для установки библиотеки, необьходимо указать путь к ее местоположению (на пример с помощью пакета sys), 
а далее импортировать.

```

```


### Пример использования библиотеки


```
# Импорт необходимых библиотек
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from autoMlBinary import AutoMlBinary

import warnings
warnings.filterwarnings("ignore")

# Пример использования 
def main():
    # getting example data from sklearn.datasets and splitting it
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)

    # creating an instance of
    automl = AutoMlBinary(
        seed=0,
        cv=5,
        n_workers=4,
        models_names=['Random_forest', 'Logistic_regression']
    )
    # during the fit() best hyperparameters for each model are searched through gridsearchs using cross-validation.
    # After that models are fitted on best hyperparameters sets and compared.
    automl.fit(X_train, y_train)
    # score/predict functions work with the best score model
    print(automl.score(X_test, y_test))
    # best model with parameters
    print(automl.show_best_model())
    predictions = automl.predict(X_test)

    print(f'Accuracy score: {sklearn.metrics.accuracy_score(y_test, predictions)}')
    # save best score model with given path
    automl.save('/home/oleg/Documents/ouput_automl')


if __name__ == '__main__':
    main()
```

## Deployment

Модели, полученные в результате работы библиотеки можно сохранить на локальную систему
для дальнейшего использования


