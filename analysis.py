import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_curve, auc, silhouette_score
from sklearn.preprocessing import label_binarize
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


def pca_f(pca, X, y, third_dimension = None):

    # Represent each predictor by the principle components
    X_r = pca.fit(X).transform(X)

    print(f"Average Silhouette width: {silhouette_score(X_r, y, random_state=1)}")

    # Percentage of variance explained for each components
    # print("explained variance ratio (first two components): %s"% str(pca.explained_variance_ratio_))

    # Create the figure and 3d axes
    fig = plt.figure(layout="constrained")

    colors = ["navy", "turquoise", "darkorange"]
    lw = 2


    if third_dimension:
        ax = fig.add_subplot(projection="3d")
    else:
        ax = fig.add_subplot()

    for color, i in zip(colors, [1, 2, 3]):
        # Filter the Xs to only the selected class
        subset = X_r[y == i]
        if third_dimension:
            # Plot the Xs by the principle components; color by label
            ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], color=color, alpha=0.4, lw=lw, label=Mapper().to_class(i))

        else:
            ax.scatter(subset[:, 0], subset[:, 1], color=color, alpha=0.4, lw=lw, label=Mapper().to_class(i))
    # Build Labels and titles
    ax.set_xlabel(f"PCA_1: {100*pca.explained_variance_ratio_[0]:.2f}%")
    ax.set_ylabel(f"PCA_2: {100*pca.explained_variance_ratio_[1]:.2f}%")
    if third_dimension:
        ax.set_zlabel(f"PCA_3: {100*pca.explained_variance_ratio_[2]:.2f}%")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of NASA dataset")
    plt.show()

def try_RFC(X, y, rfc):
    # Split the data 80/20: train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Binarize the labels for multi-class ROC AUC
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]

    # Split the binarized labels
    y_train_bin = label_binarize(y_train, classes=np.unique(y))
    y_test_bin = label_binarize(y_test, classes=np.unique(y))

    # Fit the classifier to the training data
    rfc.fit(X_train, y_train)

    # Build the predicted probabilities for the test dataset
    y_pred_prob = rfc.predict_proba(X_test)

    # Compute ROC AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotting each class
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {Mapper().to_class(i+1)} (area = {roc_auc[i]:.8f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC -- Classifying stellar objects')
    plt.legend(loc="lower right")
    plt.show()


    importances = rfc.feature_importances_
    feature_names = X.columns

    # Combine importances with feature names
    feature_importances = list(zip(feature_names, importances))

    # Sort the features by importance
    sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    print("Feature Importances:")
    for feature, importance in sorted_features:
        print(f"{feature:>9}: {importance:.4f}")



    # Calculate Accuracy for the test dataset
    acc = accuracy_score(y_test, rfc.predict(X_test))
    print(f"Accuracy for single iteration: {acc}")




# def make_descriptive_tables(df:pd.DataFrame):
    
#     # print()
#     # for c in range (1,4):
#     #     print(df[df["class"] == Mapper().to_class(c)].describe(percentiles=[]))



def load_data(fpath, samplesize : int | None = None):

    # Load the data to a pd.DataFrame
    full_df = pd.read_csv(fpath + "\\star_classification.csv")

    # Filter out erroneous data
    full_df = full_df[full_df.u>-999].drop(columns=["rerun_ID"])

    full_df.drop(columns = ["cam_col", "run_ID", "field_ID", "fiber_ID", "obj_ID", "plate", "MJD", "alpha", "delta", "spec_obj_ID"], inplace=True) #"spec_obj_ID"

    # Sample the DataFrame (speed up runtime)
    if samplesize != None:
        full_df = full_df.sample(samplesize, random_state=1)

    # Build the predictor features
    features = full_df.drop(columns=["class"])

    # Build the class labels
    labels = full_df["class"].apply(lambda x: Mapper().to_int(x)).ravel()

    return features, labels, full_df


def main():
    # ----------------------------------------------------------------------------- #

    # Define the working directory
    path = os.path.dirname(os.path.abspath(__file__))

    # Get the Predictors and labels
    X, y, full_df = load_data(path)

    # Show descriptive statistics of the classes
    # make_descriptive_tables(full_df)

    # Standardize the data
    X = standardize_input(X)

    # ----------------------------------------------------------------------------- #

    # Build the Principle Components Engine
    pca = PCA(n_components=3,random_state=1)

    # Visualize the primary Principle components and their relation to the classes
    pca_f(pca, X, y)
        # We notice there are not clear clusters of classes.
        # Things like KMeans or adjacent clustering techniques will not work well. 

    # ----------------------------------------------------------------------------- #

    # Create the Random Forest Classifier Model
    rfc = RandomForestClassifier(random_state=1)

    # Apply The RFC to the data.
    try_RFC(X, y, rfc)

    # ----------------------------------------------------------------------------- #
main()
