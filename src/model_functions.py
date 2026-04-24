import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


MODEL_PATH = "../models/our_model.pkl"
DATA_PATH = "../data/10_points.csv"


def predict_value(x):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    x_array = np.array([[float(x)]])
    prediction = model.predict(x_array)

    return float(prediction[0])


def update_model(x, y):
    x = float(x)
    y = float(y)

    df = pd.read_csv(DATA_PATH)

    new_row = pd.DataFrame({"x": [x], "y": [y]})
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(DATA_PATH, index=False)

    X = df["x"].values.reshape(-1, 1)
    y_values = df["y"].values

    model = LinearRegression()
    model.fit(X, y_values)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    return "Model został zaktualizowany"