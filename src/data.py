import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TEST_SIZE, RANDOM_SEED, TARGET


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def split_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    return X_train, X_test, y_train, y_test
