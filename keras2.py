from keras import losses
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adamax
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

from helpers import *


def main():
    train_data, test_data, train_labels, test_labels = get_iris_data(train_size=0.7)

    train_data, test_data = scale(MinMaxScaler(), train_data, test_data)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    output_node = int(len(train_labels[0]))
    input_node = int(len(train_data[0]))

    model = create_model(input_node, output_node)
    model.summary()

    test_pred_sgd, history_sgd = train_and_predict(model,
                                                   SGD(learning_rate=0.1),
                                                   train_data, train_labels, test_data)

    test_pred_adam, history_adam = train_and_predict(model,
                                                    Adam(learning_rate=0.1),
                                                    train_data, train_labels, test_data)

    print_result("GRADIENT DESCENT OPTIMIZER",
                 test_pred_sgd, test_labels, axis=1)

    print_result("ADAPTIVE MOMENTUM OPTIMIZER",
                 test_pred_adam, test_labels, axis=1)

    show_plot(history_sgd, history_adam, "accuracy", ["sgd", "adam"])
    show_plot(history_sgd, history_adam, "loss", ["sgd", "adam"])


def create_model(input_node, output_node):
    model = Sequential(name="project")
    model.add(Dense(input_node, activation="sigmoid",
              name="input", input_shape=(input_node,)))
    model.add(Dense(input_node * 2, activation="sigmoid", name="hidden"))
    model.add(Dense(output_node, activation="sigmoid",  name="output"))
    return model


def train_and_predict(model, optimizer, train_data, train_labels, test_data):
    model.compile(loss=losses.BinaryCrossentropy(),
                  metrics="accuracy", optimizer=optimizer)

    history = model.fit(train_data, train_labels, epochs=50, batch_size=1,
                        validation_split=0.1, use_multiprocessing=True, verbose=1)

    test_pred = model.predict(test_data, use_multiprocessing=True, verbose=1)

    return test_pred, history


main()
