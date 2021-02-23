import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)


alphabet = "abcdefghijklmnopqrstuvwxyz"

# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


def one_hot_encode(data_array):

    print(data_array)

    integer_encoded = [char_to_int[char] for char in data_array]

    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)

    return onehot_encoded


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

train, test = train_test_split(df, test_size=0.3)

print(df.dtypes)

print(df.shape)

print(train.head())

train_features = train.copy()

train_features = pd.DataFrame(data=train_features, columns=column_names)

train_features.pop("class")

train_features_converted = pd.get_dummies(
    train_features, columns=train_features.columns
)

print(train_features_converted)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(112,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2),
    ]
)

predictions = model(np.array(train_features_converted))
print(predictions)

print(tf.nn.softmax(predictions).numpy())