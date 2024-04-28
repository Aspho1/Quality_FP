import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class Mapper:
    def __init__(self):
        self.recode_int = {"GALAXY": 1, "QSO": 2, "STAR": 3}
        self.recode_label = {1 : "GALAXY", 2 : "QSO", 3 : "STAR"}

    def to_int(self, label):
        return self.recode_int[label]
    
    def to_class(self, int_val):
        return self.recode_label[int_val]

def standardize_input(X) -> pd.DataFrame:
    # Scale each datapoint to be a fraction of standard deviations away from the mean
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    return X


def pca_f(pca, X, y):

    # Represent each predictor by the principle components
    X_r = pca.fit(X).transform(X)

    # Percentage of variance explained for each components
    # print("explained variance ratio (first two components): %s"% str(pca.explained_variance_ratio_))

    # Create the figure and 3d axes
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    colors = ["navy", "turquoise", "darkorange"]
    lw = 2

    for color, i in zip(colors, [1, 2, 3]):
        # Filter the Xs to only the selected class
        subset = X_r[y == i]

        # Plot the Xs by the principle components; color by label
        ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], color=color, alpha=0.4, lw=lw, label=Mapper().to_class(i))

    # Build Labels and titles
    ax.set_xlabel(f"PCA_1: {100*pca.explained_variance_ratio_[0]:.2f}%")
    ax.set_ylabel(f"PCA_2: {100*pca.explained_variance_ratio_[1]:.2f}%")
    ax.set_zlabel(f"PCA_3: {100*pca.explained_variance_ratio_[2]:.2f}%")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of NASA dataset")
    plt.show()

def try_RFC(X, y, rfc):
    # Split the data 80/20 : train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Fit the classifier to the training data
    rfc.fit(X_train, y_train)

    # Build the predicted labels for the test dataset
    y_pred = rfc.predict(X_test)

    # Calculate Accuracy for the test dataset
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for single iteration: ", acc)







def load_data(fpath, samplesize : int | None = None):

    # Load the data to a pd.DataFrame
    full_df = pd.read_csv(fpath + "\\star_classification.csv")

    # Filter out erroneous data
    full_df = full_df[full_df.u>-999]

    # Sample the DataFrame (speed up runtime)
    if samplesize != None:
        full_df = full_df.sample(samplesize, random_state=1)

    # Build the predictor features
    features = full_df.drop(columns=["class","rerun_ID"])

    # Build the class labels
    labels = full_df["class"].apply(lambda x: Mapper().to_int(x)).ravel()

    return features, labels


def main():

    # Define the working directory
    path = os.path.dirname(os.path.abspath(__file__))

    # Get the Predictors and labels
    X, y = load_data(path)

    # # Print the first 5 rows of the predictors and the labels
    # print(X.head())
    # print(y[:5])

    # Standardize the data
    X = standardize_input(X)

    # Visualize the primary Principle components and their relation to the classes
    pca = PCA(n_components=3,random_state=1)
    pca_f(pca, X, y)
        # We notice there are not clear clusters of classes.
        # Things like KMeans or adjacent clustering techniques will not work well. 


    # Create the Random Forest Classifier Model
    rfc = RandomForestClassifier(random_state=1)

    # Apply The RFC to the data.
    try_RFC(X, y, rfc)


main()
