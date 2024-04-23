import pandas as pd
import os

from sklearn.metrics import accuracy_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline

def try_RFC_KFolds(X, y, rfc):

    kfolds=StratifiedKFold(n_splits=10)

    pipe = make_pipeline(rfc, )

    fold_num = 1
    for train_index, test_index in kfolds.split(X,y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pipe.fit(X_train,y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        # cal = calinski_harabasz_score(y_test, y_pred)
        print(f"Accuracy for fold {fold_num}: ", acc)
        # print(f"Accuracy for fold {fold_num}: ", cal)
        fold_num += 1


def try_RFC(X, y, rfc):

    pipe = make_pipeline(rfc)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # cal = calinski_harabasz_score(y_test, y_pred)
    print(f"Accuracy for single iteration: ", acc)
    # print(f"Accuracy for fold {fold_num}: ", cal)






def grid_search_parameters(X,y, rfc):
    print("performing a grid_search for optimal RFC parameters based on AUC Score")

    param_grid = {

        'n_estimators': [100, 200, 300], 
        # 'n_estimators': [100 , 200], 
        'max_features': ['sqrt', None],
        # 'max_features': ['sqrt'],
        # 'bootstrap': [True],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_leaf': [1, 2, 3],
        # 'min_samples_leaf': [1],
    }

    print(f"    Included parameters: {param_grid}")
    grid_search = GridSearchCV(rfc, param_grid, scoring='rand_score', cv=5)
    grid_search.fit(X, y)

    # Get the best model and AUC score
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_

    # Output the best score
    print(best_model)
    print(f"best roc_AUC: {best_score:5f}")


def load_data(fpath) -> pd.DataFrame:
    full_df = pd.read_csv(fpath + "\\star_classification.csv")
    features = full_df.drop(columns = "class")
    labels = full_df["class"].map({"GALAXY":1, "QSO":2, "STAR":3}).ravel()

    return features, labels

def main():
    path = os.path.dirname(os.path.abspath(__file__))
    X, y = load_data(path)
    print(X.head())
    print(y[:5])

    rfc = RandomForestClassifier(random_state=1)

    grid_search_parameters(X,y,rfc)

    # try_RFC_KFolds(X,y,rfc)

    try_RFC(X,y,rfc)





main()