import argparse
import os
import joblib
import numpy as np
from azureml.core import Dataset, Workspace
from azureml.core.run import Run
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest.")
    parser.add_argument('--max_depth', type=int, default=None, help="The maximum depth of the tree.")

    args = parser.parse_args()
    ws = Workspace.from_config()
    run = Run.get_context()

    run.log("n_estimators:", np.float(args.n_estimators))
    run.log("max_depth:", np.int(args.max_depth))

    # Load registered dataset
    iris_dataset = Dataset.get_by_name(workspace=ws, name='iris_dataset')
    iris_df = iris_dataset.to_pandas_dataframe()

    X, y = iris_df.drop(columns=['target']), iris_df['target']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Model evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    run.log("Accuracy", np.float(accuracy))

    # Save the model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=clf, filename='outputs/hyperdrive_model.pkl')

if __name__ == '__main__':
    main()
