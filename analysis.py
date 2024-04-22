import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier


def try_RFC(dataset):
    rfc = RandomForestClassifier
    rfc


def load_data(fpath) -> pd.DataFrame:
    full_df = pd.read_csv(fpath + "\\star_classification.csv")
    features = full_df.drop(columns = "class")
    labels = full_df[["class"]]
    return features, labels

def main():
    path = os.path.dirname(os.path.abspath(__file__))
    X, y = load_data(path)
    print(X.head())
    print(y.head())



main()