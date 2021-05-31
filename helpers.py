import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from termcolor import colored
import matplotlib.pyplot as plt


def get_iris_data(train_size):

    iris = load_iris()

    iris.data, iris.target = shuffle(iris.data, iris.target)

    return train_test_split(iris.data, iris.target, train_size=train_size)


def scale(scaler, train_data, test_data):
    scaler.fit(np.concatenate((train_data, test_data)))

    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data


def print_result(title, test_pred, test_labels, axis=0):
    print("\n", colored(title, "green"))

    if(axis > 0):
        test_pred_argmax = np.argmax(test_pred, axis=axis)
        test_labels_argmax = np.argmax(test_labels, axis=axis)
    else:
        test_pred_argmax = test_pred
        test_labels_argmax = test_labels

    print(colored("\nAccuracy Score : ", "red"))
    print(accuracy_score(test_pred_argmax, test_labels_argmax))

    cm = confusion_matrix(test_pred_argmax, test_labels_argmax)

    print(colored("\nConfussion Matrix : \n", "red"))
    print(
        "\t 0\t 1\t 2\n",
        "0\t", cm[0][0], "\t", cm[0][1], "\t", cm[0][2], "\n",
        "1\t", cm[1][0], "\t", cm[1][1], "\t", cm[1][2], "\n",
        "2\t", cm[2][0], "\t", cm[2][1], "\t", cm[2][2], "\n",
    )

    print(colored("\nClassification Report : \n", "red"))
    print(classification_report(test_pred_argmax, test_labels_argmax))


def show_plot(history1, history2, key, labels):
    print(history1)
    plt.plot(history1.history[key])
    plt.plot(history2.history[key])
    plt.title("model " + key)
    plt.ylabel(key)
    plt.xlabel("epoch")
    plt.legend(labels, loc="upper left")
    plt.show()
