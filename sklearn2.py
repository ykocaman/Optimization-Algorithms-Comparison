from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skopt import BayesSearchCV

from helpers import *

train_data, test_data, train_labels, test_labels = get_iris_data(
    train_size=0.7)

train_data, test_data = scale(StandardScaler(), train_data, test_data)

mlp = MLPClassifier(
    solver='adam',
    activation='relu',
    hidden_layer_sizes=(8, ),
    random_state=1,
    max_iter=10,
    n_iter_no_change=1,
    learning_rate_init=.01,
    verbose=10,
)

bs = BayesSearchCV(
    SVC(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),
        'kernel': ['linear', 'poly', 'rbf'],
    },
    n_iter=10,
    cv=2,
    random_state=1,
    verbose=10
)

history_mlp = mlp.fit(train_data, train_labels)

history_bs = bs.fit(train_data, train_labels)

test_pred_mlp = mlp.predict(test_data)

test_pred_bs = bs.predict(test_data)

print_result("GRADIENT DESCENT OPTIMIZER", test_pred_mlp, test_labels)

print_result("BAYESIAN SEARCH (Hyperparameter) OPTIMIZER", test_pred_bs, test_labels)
