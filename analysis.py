import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_curve, auc, silhouette_score
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
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

    # print(f"Average Silhouette width: {silhouette_score(X_r, y, random_state=1)}")

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

def grid_search_parameters(X,y, rfc):
    rfc
    print("performing a grid_search for optimal RFC parameters")

    # param_grid = {'n_estimators': [100, 200, 300], 'max_features': ['sqrt', None], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2]}
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    print(f"    Included parameters: {param_grid}")
    grid_search = GridSearchCV(rfc, param_grid, scoring='f1_micro', cv=4, n_jobs=-1, verbose=1)
    # grid_search = RandomizedSearchCV(rfc, param_grid, scoring='f1_micro', cv=4, n_jobs=-1, n_iter=50, verbose=1)
    grid_search.fit(X, y)

    # Get the best model and AUC score
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_

    # Output the best score
    print(best_model)
    print(f"best roc_AUC: {best_score:5f}")



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

    # Plot each ROC curve 
    fig, ax = plt.subplots(layout="constrained")

    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f'ROC of {Mapper().to_class(i+1)} (AUC = {100*roc_auc[i]:.2f}%)')

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    fig.suptitle('Multi-class ROC -- Classifying Stellar Objects')

    importances = rfc.feature_importances_
    feature_names = X.columns

    # Combine importances with feature names
    feature_importances = list(zip(feature_names, importances))

    # Sort the features by importance
    sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_features)

    fig, ax = plt.subplots(layout="constrained")
    ax.bar(features, importances, color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importances in Random Forest Classifier')


    # Calculate performance metrics for the test set
    acc = accuracy_score(y_test, rfc.predict(X_test))
    print(f"Accuracy for single iteration: {acc}")
    print(f"Overall Average ROCAUC: {np.mean([i for i in roc_auc.values()]):.4f}")

# def make_descriptive_tables(df:pd.DataFrame):
    
#     # print()
#     # for c in range (1,4):
#     #     print(df[df["class"] == Mapper().to_class(c)].describe(percentiles=[]))



def load_data(fpath, samplesize : int | None = None):

    # Load the data to a pd.DataFrame
    full_df = pd.read_csv(fpath + "\\star_classification.csv")

    # Filter out erroneous data
    # full_df = full_df[full_df.u>-999].sample(10000)
    full_df = full_df[full_df.u>-999]


    # Got rid of meta columns. Only have columns which actually report information on the luminance of the stellar object remaining. 
    full_df.drop(columns = ["rerun_ID", "cam_col", "run_ID", "field_ID", "fiber_ID", "obj_ID", "plate", "MJD", "alpha", "delta", "spec_obj_ID"], inplace=True) #"spec_obj_ID"

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
    # Grid searched for AUC
    rfc = RandomForestClassifier(criterion='entropy', max_features=None,
                       min_samples_leaf=2, n_estimators=400, random_state=1)
    
    # Grid searched for accuracy NOT DONE YET
    # rfc = RandomForestClassifier(random_state=1)

    print("Average Silhouette width: 0.0946 (this is low, not well suited for clustering)")

    # Gridsearch Parameters
    # grid_search_parameters(X,y,rfc)

    # Apply The RFC to the data.
    try_RFC(X, y, rfc)

    plt.show()

    # ----------------------------------------------------------------------------- #
main()
