import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time

# learning_rate=0.0001

# from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical


def split_features_labels(dataset):
    features = dataset.copy()

    features.pop("class_e")
    features.pop("class_p")

    labels = dataset[["class_e", "class_p"]].copy()

    return features, labels


def one_hot_encode(dataset):

    return pd.get_dummies(dataset, columns=dataset.columns)


def train_model(x, y, epochs, hidden_layer):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(112,)),
            tf.keras.layers.Activation(activation="sigmoid"),
        ]
    )

    if hidden_layer:
        model.add(tf.keras.layers.Dense(10, activation="sigmoid", name="hidden_layer1"))

    model.add(tf.keras.layers.Dense(2))

    # model.add(tf.keras.layers.Softmax())

    predictions = model(train_features)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        loss=loss_fn,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    start_time = time.time()

    history = model.fit(x=x, y=y, epochs=epochs)

    exec_time = str(round((time.time() - start_time), 2))

    print("\nTraining took:\t" + exec_time + " seconds\n")

    model.evaluate(test_features, test_labels, verbose=2)

    return history


# column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
column_names = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

df = pd.read_csv("mushrooms.csv", names=column_names)

df.pop("stalk-root")

df = one_hot_encode(df)

# print(df)

train, test = train_test_split(df, test_size=0.3)

train_features, train_labels = split_features_labels(train)
test_features, test_labels = split_features_labels(test)

train_features = np.array(train_features)
train_labels = np.array(train_labels)

test_features = np.array(test_features)
test_labels = np.array(test_labels)

# train_features = one_hot_encode(train_features)
# train_labels = one_hot_encode(train_labels)
# test_features = one_hot_encode(test_features)
# test_labels = one_hot_encode(test_labels)

epochs = 50

history_no_hidden_nodes = train_model(train_features, train_labels, epochs, False)
history_hidden_nodes = train_model(train_features, train_labels, epochs, True)

plt.plot(history_no_hidden_nodes.history["accuracy"], label="No hidden layer")
plt.plot(history_hidden_nodes.history["accuracy"], label="One hidden layers")

plt.legend()

plt.title("Accuracy per epoch of training")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")

plt.savefig("accuracy")