import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def std(X):

    for i in range(len(X.columns)):
        target_column = X.columns[i]
        X[target_column] = (X[target_column] - X[target_column].mean()) / X[target_column].std()
    return X


def pca_f(X, y):

    target_names = y

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    # Percentage of variance explained for each components
    print("explained variance ratio (first two components): %s"% str(pca.explained_variance_ratio_))


    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2

    for color, i, target_name in zip(colors, [1, 2, 3], target_names):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.4, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of NASA dataset")
    plt.show()

def try_RFC(X, y, rfc):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # cal = calinski_harabasz_score(y_test, y_pred)
    print(f"Accuracy for single iteration: ", acc)
    # print(f"Accuracy for fold {fold_num}: ", cal)





def load_data(fpath):
    full_df = pd.read_csv(fpath + "\\star_classification.csv")
    full_df = full_df[full_df.u>-999]
    features = full_df.drop(columns=["class","rerun_ID"])
    labels = full_df["class"].map({"GALAXY": 1, "QSO": 2, "STAR": 3}).ravel()

    return features, labels


def main():
    path = os.path.dirname(os.path.abspath(__file__))
    X, y = load_data(path)
    print(X.head())
    print(y[:5])

    rfc = RandomForestClassifier(random_state=1)

   # try_RFC(X, y, rfc)

    X = std(X)
    pca_f(X, y)
    print(X.columns)


main()
